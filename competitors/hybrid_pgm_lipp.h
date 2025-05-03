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
        : dp_index_(params), lipp_index_(params),
          insert_buffer_a_(), insert_buffer_b_(),
          active_buffer_(&insert_buffer_a_), flush_buffer_(&insert_buffer_b_),
          flushing_(false), insert_count_(0) {
        flush_threshold_ = params.empty() ? 100000 : params[0];
    }

    ~HybridPGMLIPP() {
        wait_for_flush();
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
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            active_buffer_->emplace_back(data);
            insert_count_++;
        }

        dp_index_.Insert(data, thread_id);

        if (insert_count_ >= flush_threshold_ && !flushing_.exchange(true)) {
            // Swap buffers under mutex and launch flush thread
            {
                std::lock_guard<std::mutex> guard(buffer_mutex_);
                std::swap(active_buffer_, flush_buffer_);
                insert_count_ = 0;
            }
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }
    }

    std::string name() const { return "HybridPGMLIPP"; }

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
        std::vector<KeyValue<KeyType>> local_snapshot;
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            local_snapshot.swap(*flush_buffer_);
        }

        for (const auto& kv : local_snapshot) {
            lipp_index_.Insert(kv, 0);
        }

        dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
        flushing_ = false;
    }

    void wait_for_flush() {
        if (flush_thread_.joinable()) {
            flush_thread_.join();
        }
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    std::vector<KeyValue<KeyType>> insert_buffer_a_;
    std::vector<KeyValue<KeyType>> insert_buffer_b_;
    std::vector<KeyValue<KeyType>>* active_buffer_;
    std::vector<KeyValue<KeyType>>* flush_buffer_;
    std::mutex buffer_mutex_;

    std::atomic<bool> flushing_;
    size_t insert_count_;
    size_t flush_threshold_;
    std::thread flush_thread_;
};
