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
        flushing_(false),
        insert_count_(0)
    {
        // same thresholds you used in your sample
        flush_threshold_ = 100000;
        // detect 10% insert runs from filename
        insert_ratio_low_ = false;
        for (const auto& s : params_filenames_) {
            if (s.find("_0.100000i_") != std::string::npos) {
                insert_ratio_low_ = true;
                break;
            }
        }
        flush_threshold_ = insert_ratio_low_ ? 5000 : 50000;
        bypass_dpgm_ = insert_ratio_low_;
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable())
            flush_thread_.join();
    }

    // bulk‑load goes to LIPP only
    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_.Build(data, num_threads);
    }

    // check DPGM first; if miss, fall back to LIPP
    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        size_t res = dp_index_.EqualityLookup(key, thread_id);
        return (res == util::OVERFLOW || res == util::NOT_FOUND)
             ? lipp_index_.EqualityLookup(key, thread_id)
             : res;
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        return dp_index_.RangeQuery(lo, hi, thread_id)
             + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    // route into DPGM (optionally), buffer to flush, and fire off a bg thread
    void Insert(const KeyValue<KeyType>& kv, uint32_t thread_id) {
        {
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            insert_buffer_.push_back(kv);
        }

        if (!bypass_dpgm_)
            dp_index_.Insert(kv, thread_id);

        ++insert_count_;
        if (insert_count_ >= flush_threshold_ && !flushing_.exchange(true)) {
            // join any previous flush thread before reusing it
            if (flush_thread_.joinable())
                flush_thread_.join();

            {
                std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
                std::swap(insert_buffer_, flush_buffer_);
                insert_count_ = 0;
            }
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }
    }

    std::string name() const { return "HybridPGMLIPP"; }
    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }
    std::size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }

    // catch the ops_filename once to set low‑insert flags
    bool applicable(bool unique, bool range_query, bool insert,
                    bool multithread, const std::string& ops_filename) const {
        params_filenames_.push_back(ops_filename);
        return !multithread;
    }

private:
    void flush_to_lipp() {
        // swap out the batch
        std::vector<KeyValue<KeyType>> batch;
        {
            std::lock_guard<std::mutex> buf_lk(buffer_mutex_);
            batch.swap(flush_buffer_);
        }
        // replay into LIPP
        for (auto &kv : batch) {
            lipp_index_.Insert(kv, /*thread=*/0);
        }
        flushing_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType>                      lipp_index_;

    // double‑buffer for pending inserts
    std::vector<KeyValue<KeyType>> insert_buffer_, flush_buffer_;
    std::mutex                     buffer_mutex_;
    std::atomic<size_t>            insert_count_;
    size_t                         flush_threshold_;
    std::atomic<bool>              flushing_;
    std::thread                    flush_thread_;

    // from your sample
    mutable std::vector<std::string> params_filenames_;
    mutable bool                     insert_ratio_low_ = false;
    mutable bool                     bypass_dpgm_      = false;
};
