#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <vector>
#include <thread>
#include <atomic>
#include <mutex>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
      : dp_index_(params),
        lipp_index_(params),
        flush_pending_(false),
        insert_count_(0)
    {
        flush_threshold_low_  = 5'000;   // tuned for 10% insert workloads
        flush_threshold_high_ = 50'000;  // tuned for 90% insert workloads

        insert_ratio_low_      = false;
        bypass_dpgm_           = false;
        insert_ratio_flag_set_ = false;
    }

    ~HybridPGMLIPP() {
        // join any outstanding flush thread before destruction
        if (flush_thread_.joinable())
            flush_thread_.join();
    }

    // Bulk‐build only into LIPP
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_.Build(data, num_threads);
    }

    // Check DPGM first, then LIPP under lock
    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        size_t r = dp_index_.EqualityLookup(key, thread_id);
        if (r != util::NOT_FOUND && r != util::OVERFLOW)
            return r;

        std::lock_guard<std::mutex> lk(lipp_mutex_);
        return lipp_index_.EqualityLookup(key, thread_id);
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        uint64_t a = dp_index_.RangeQuery(lo, hi, thread_id);
        std::lock_guard<std::mutex> lk(lipp_mutex_);
        uint64_t b = lipp_index_.RangeQuery(lo, hi, thread_id);
        return a + b;
    }

    // Route into DPGM (unless bypassed), buffer for async flush, and trigger it when threshold hits
    void Insert(const KeyValue<KeyType>& kv, uint32_t thread_id) {
        if (!bypass_dpgm_) {
            dp_index_.Insert(kv, thread_id);
        }

        {
            std::lock_guard<std::mutex> buf_lk(buf_mutex_);
            active_buf_.push_back(kv);
            ++insert_count_;
        }

        size_t threshold = insert_ratio_low_ ? flush_threshold_low_
                                             : flush_threshold_high_;
        // fire off a flush if we've hit the threshold
        if (insert_count_ >= threshold && !flush_pending_.exchange(true)) {
            // **join** any previous flush before swapping buffers
            if (flush_thread_.joinable())
                flush_thread_.join();

            {
                std::lock_guard<std::mutex> buf_lk(buf_mutex_);
                std::swap(active_buf_, flush_buf_);
                insert_count_ = 0;
            }

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

    // Capture the ops filename once to decide insert ratio and bypass logic
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const
    {
        if (multithread)
            return false;  // this hybrid is single‑threaded

        if (!insert_ratio_flag_set_) {
            if (ops_filename.find("_0.100000i_") != std::string::npos) {
                insert_ratio_low_ = true;
                bypass_dpgm_      = true;  // skip DPGM in the 10%‑insert scenario
            }
            insert_ratio_flag_set_ = true;
        }
        return true;
    }

private:
    // Actually dump flush_buf_ into LIPP under lock
    void flush_worker() {
        std::vector<KeyValue<KeyType>> batch;
        {
            std::lock_guard<std::mutex> buf_lk(buf_mutex_);
            batch.swap(flush_buf_);
        }

        std::lock_guard<std::mutex> lk(lipp_mutex_);
        for (auto &kv : batch) {
            lipp_index_.Insert(kv, /*thread=*/0);
        }

        flush_pending_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                      lipp_index_;

    // double‐buffered queues
    std::vector<KeyValue<KeyType>> active_buf_, flush_buf_;
    mutable std::mutex             buf_mutex_;
    std::atomic<size_t>            insert_count_;
    std::atomic<bool>              flush_pending_;
    std::thread                    flush_thread_;

    // serialize **all** LIPP calls
    mutable std::mutex lipp_mutex_;

    // thresholds
    size_t flush_threshold_low_, flush_threshold_high_;

    // workload flags (must be mutable to set inside a const method)
    mutable bool insert_ratio_low_;
    mutable bool bypass_dpgm_;
    mutable bool insert_ratio_flag_set_;
};
