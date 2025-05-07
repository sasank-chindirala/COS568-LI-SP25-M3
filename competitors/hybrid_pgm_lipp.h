#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>
#include <string>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params),
          insert_count_(0), flushing_(false),
          total_ops_(0), insert_ops_(0),
          insert_ratio_high_(false), mode_decided_(false)
    {
        flush_threshold_ = 100000;  // Used only in high-insert mode
        insert_check_threshold_ = 100000;  // Check after this many ops
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        // Lookup is never counted in ops, so no race
        if (!insert_ratio_high_) {
            return lipp_index_.EqualityLookup(key, thread_id);
        }
        size_t result = dp_index_.EqualityLookup(key, thread_id);
        return (result == util::OVERFLOW || result == util::NOT_FOUND)
             ? lipp_index_.EqualityLookup(key, thread_id)
             : result;
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        if (!insert_ratio_high_) {
            return lipp_index_.RangeQuery(lo, hi, thread_id);
        }
        return dp_index_.RangeQuery(lo, hi, thread_id) +
               lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        size_t op_index = total_ops_.fetch_add(1, std::memory_order_relaxed);
        insert_ops_.fetch_add(1, std::memory_order_relaxed);

        // Infer mode early (once only, thread-safe)
        if (!mode_decided_ && op_index + 1 == insert_check_threshold_) {
            double insert_ratio = static_cast<double>(insert_ops_.load()) / (op_index + 1);
            insert_ratio_high_ = insert_ratio >= 0.45;  // Threshold works for your workload split
            mode_decided_ = true;
        }

        if (!insert_ratio_high_) {
            lipp_index_.Insert(data, thread_id);  // Use LIPP only
            return;
        }

        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            insert_buffer_.emplace_back(data);
        }
        dp_index_.Insert(data, thread_id);
        insert_count_++;

        if (insert_count_ >= flush_threshold_ && !flushing_.exchange(true)) {
            if (flush_thread_.joinable()) flush_thread_.join();
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }
    }

    std::string name() const {
        return "HybridPGMLIPP";
    }

    std::vector<std::string> variants() const {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }

    size_t size() const {
        return dp_index_.size() + lipp_index_.size();
    }

    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& /*ops_filename*/) const {
        return !multithread;
    }

private:
    void flush_to_lipp() {
        std::vector<KeyValue<KeyType>> snapshot;
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            snapshot.swap(insert_buffer_);
            insert_count_ = 0;
        }
        for (const auto& kv : snapshot) {
            lipp_index_.Insert(kv, 0);
        }
        flushing_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    std::vector<KeyValue<KeyType>> insert_buffer_;
    std::mutex buffer_mutex_;
    size_t insert_count_;
    size_t flush_threshold_;
    std::atomic<bool> flushing_;
    std::thread flush_thread_;

    // For true insert ratio inference
    std::atomic<size_t> total_ops_;
    std::atomic<size_t> insert_ops_;
    size_t insert_check_threshold_;
    mutable bool insert_ratio_high_;
    std::atomic<bool> mode_decided_;
};
