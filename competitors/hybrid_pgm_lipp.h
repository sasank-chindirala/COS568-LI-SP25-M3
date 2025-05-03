#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params), insert_count_(0), flushing_(false),
          active_buffer_(&buffer_a_), inactive_buffer_(&buffer_b_) {
        // Dynamic tuning based on insert ratio if provided
        flush_threshold_ = params.empty() ? 100000 : params[0];
        if (flush_threshold_ == 1) flush_threshold_ = 20000; // Insert-heavy tuning
        else if (flush_threshold_ == 9) flush_threshold_ = 100000; // Lookup-heavy tuning
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        size_t result = dp_index_.EqualityLookup(key, thread_id);
        return (result == util::OVERFLOW || result == util::NOT_FOUND)
            ? lipp_index_.EqualityLookup(key, thread_id)
            : result;
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        return dp_index_.RangeQuery(lo, hi, thread_id) + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        // Conditional routing: if dp_index_ size too big, insert directly to LIPP
        if (dp_index_.size() > max_dp_size_bytes_) {
            lipp_index_.Insert(data, thread_id);
            return;
        }

        // Lock buffer to insert
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            active_buffer_->emplace_back(data);
        }

        dp_index_.Insert(data, thread_id);
        size_t count = ++insert_count_;

        // Initiate async flush if threshold crossed and not already flushing
        if (count >= flush_threshold_ && !flushing_.exchange(true)) {
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
                    const std::string& ops_filename) const {
        return !multithread;
    }

private:
    void flush_to_lipp() {
        std::vector<KeyValue<KeyType>>* local_buffer;

        // Double-buffering swap
        {
            std::lock_guard<std::mutex> lock(buffer_mutex_);
            std::swap(active_buffer_, inactive_buffer_);
            local_buffer = inactive_buffer_;
            insert_count_ = 0;
        }

        for (const auto& kv : *local_buffer) {
            lipp_index_.Insert(kv, 0);
        }

        local_buffer->clear();
        flushing_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    std::vector<KeyValue<KeyType>> buffer_a_;
    std::vector<KeyValue<KeyType>> buffer_b_;
    std::vector<KeyValue<KeyType>>* active_buffer_;
    std::vector<KeyValue<KeyType>>* inactive_buffer_;
    std::mutex buffer_mutex_;

    std::atomic<size_t> insert_count_;
    size_t flush_threshold_;
    const size_t max_dp_size_bytes_ = 64ull * 1024 * 1024;  // 64MB cap for DPGM growth

    std::thread flush_thread_;
    std::atomic<bool> flushing_;
};
