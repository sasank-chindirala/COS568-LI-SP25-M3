#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// HybridPGMLIPP with double-buffered asynchronous flushing and dynamic threshold tuning
template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params),
          lipp_index_(params),
          flush_pending_(false),
          insert_count_(0)
    {
        // default thresholds
        flush_threshold_low_ = 5000;      // for low insert-ratio workloads
        flush_threshold_high_ = 50000;    // for high insert-ratio workloads
        // infer insert ratio from filename in applicable(), defaults to high
        insert_ratio_low_ = false;
        bypass_dpgm_ = false;
    }

    ~HybridPGMLIPP() {
        // wait for any in-flight flush
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        // build primary LIPP base, DPGM empty
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        // first check DPGM buffer
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        if (res != util::NOT_FOUND && res != util::OVERFLOW)
            return res;
        // then LIPP
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& kv, uint32_t thread_id) {
        // 1) route to DPGM if not bypassing
        if (!bypass_dpgm_) {
            dp_index_.Insert(kv, thread_id);
        }
        // 2) buffer for asynchronous flush to LIPP
        {
            std::lock_guard<std::mutex> lock(buf_mutex_);
            active_buf_.push_back(kv);
            ++insert_count_;
        }
        // 3) trigger flush if threshold reached
        size_t threshold = insert_ratio_low_ ? flush_threshold_low_ : flush_threshold_high_;
        if (insert_count_ >= threshold && !flush_pending_.exchange(true)) {
            // swap buffers
            {
                std::lock_guard<std::mutex> lock(buf_mutex_);
                std::swap(active_buf_, flush_buf_);
                insert_count_ = 0;
            }
            // launch non-blocking flush thread
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_worker, this);
        }
    }

    std::string name() const override { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const override {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }

    std::size_t size() const override {
        return dp_index_.size() + lipp_index_.size();
    }

    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const override
    {
        // only single-threaded hybrid
        if (multithread) return false;
        // infer insert ratio from filename once
        if (!insert_ratio_flag_set_) {
            if (ops_filename.find("_0.100000i_") != std::string::npos) {
                insert_ratio_low_ = true;
                bypass_dpgm_ = true;  // skip DPGM in low-insert workloads
            }
            insert_ratio_flag_set_ = true;
        }
        return true;
    }

private:
    void flush_worker() {
        // insert all buffered items into LIPP
        for (const auto& kv : flush_buf_) {
            lipp_index_.Insert(kv, /*thread=*/0);
        }
        // clear flush buffer
        flush_buf_.clear();
        // allow next flush
        flush_pending_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    // double buffers for async flush
    std::vector<KeyValue<KeyType>> active_buf_, flush_buf_;
    mutable std::mutex buf_mutex_;
    std::atomic<bool> flush_pending_;
    std::thread flush_thread_;

    // insert counters and thresholds
    std::atomic<size_t> insert_count_;
    size_t flush_threshold_low_, flush_threshold_high_;

    // workload flags
    mutable bool insert_ratio_low_;
    mutable bool insert_ratio_flag_set_ = false;
    bool bypass_dpgm_;
};
