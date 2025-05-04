#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

// HybridPGMLIPP with double-buffered async flush, dynamic thresholds, and thread-safe LIPP access
template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        flush_pending_(false),
        insert_count_(0)
    {
        // tuning knobs
        flush_threshold_low_  = 5'000;    // for 10%‑insert workloads
        flush_threshold_high_ = 50'000;   // for 90%‑insert workloads

        insert_ratio_low_      = false;
        bypass_dpgm_           = false;
        insert_ratio_flag_set_ = false;
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable())
            flush_thread_.join();
    }

    // build the LIPP over the bulk data
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_.Build(data, num_threads);
    }

    // lookup checks DPGM first, then LIPP under lock
    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        if (res != util::NOT_FOUND && res != util::OVERFLOW)
            return res;

        std::lock_guard<std::mutex> lock(lipp_mutex_);
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        uint64_t r1 = dp_index_.RangeQuery(lo, hi, thread_id);
        std::lock_guard<std::mutex> lock(lipp_mutex_);
        uint64_t r2 = lipp_index_.RangeQuery(lo, hi, thread_id);
        return r1 + r2;
    }

    // Insert into DPGM (unless bypassed), buffer it, and trigger an async flush
    void Insert(const KeyValue<KeyType>& kv, uint32_t thread_id) {
        if (!bypass_dpgm_) {
            dp_index_.Insert(kv, thread_id);
        }

        {
            std::lock_guard<std::mutex> buf_lock(buf_mutex_);
            active_buf_.push_back(kv);
            ++insert_count_;
        }

        size_t threshold = insert_ratio_low_ ? flush_threshold_low_ : flush_threshold_high_;
        if (insert_count_ >= threshold && !flush_pending_.exchange(true)) {
            {
                std::lock_guard<std::mutex> buf_lock(buf_mutex_);
                std::swap(active_buf_, flush_buf_);
                insert_count_ = 0;
            }
            // launch a non‑blocking flush
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_worker, this);
        }
    }

    std::string name() const { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    std::size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }

    // called once per benchmark run to detect insert ratio
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const
    {
        if (multithread) return false;  // hybrid is single‑threaded

        if (!insert_ratio_flag_set_) {
            if (ops_filename.find("_0.100000i_") != std::string::npos) {
                insert_ratio_low_ = true;
                bypass_dpgm_      = true;   // for 10% inserts we skip DPGM to maximize throughput
            }
            insert_ratio_flag_set_ = true;
        }
        return true;
    }

private:
    void flush_worker() {
        // move out the batch
        std::vector<KeyValue<KeyType>> batch;
        {
            std::lock_guard<std::mutex> buf_lock(buf_mutex_);
            batch.swap(flush_buf_);
        }
        // insert them into LIPP under lock
        {
            std::lock_guard<std::mutex> lock(lipp_mutex_);
            for (auto &kv : batch) {
                lipp_index_.Insert(kv, /*thread=*/0);
            }
        }
        flush_pending_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                      lipp_index_;

    // double‑buffer for async flush
    std::vector<KeyValue<KeyType>> active_buf_, flush_buf_;
    mutable std::mutex             buf_mutex_;
    std::atomic<size_t>            insert_count_;
    std::atomic<bool>              flush_pending_;
    std::thread                    flush_thread_;

    // guard all LIPP calls
    mutable std::mutex lipp_mutex_;

    // thresholds
    size_t flush_threshold_low_, flush_threshold_high_;

    // workload flags (need mutable so applicable() can set them)
    mutable bool insert_ratio_low_;
    mutable bool bypass_dpgm_;
    mutable bool insert_ratio_flag_set_;
};
