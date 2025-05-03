#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

template<class KeyType, class SearchClass, size_t pgm_error>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : dp_index_(params), lipp_index_(params), insert_count_(0), flushing_(false) {
        flush_threshold_ = params.empty() ? 100000 : params[0];
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) override {
        return lipp_index_.Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const override {
        size_t result = dp_index_.EqualityLookup(key, thread_id);
        return (result == util::OVERFLOW || result == util::NOT_FOUND)
                   ? lipp_index_.EqualityLookup(key, thread_id)
                   : result;
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const override {
        return dp_index_.RangeQuery(lo, hi, thread_id) + lipp_index_.RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) override {
        if (flushing_.load()) {
            lipp_index_.Insert(data, thread_id);
            return;
        }

        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.emplace_back(data);
        }

        dp_index_.Insert(data, thread_id);
        insert_count_++;

        if (insert_count_ >= flush_threshold_ && !flushing_.exchange(true)) {
            if (flush_thread_.joinable()) flush_thread_.join();
            flush_thread_ = std::thread(&HybridPGMLIPP::flush_to_lipp, this);
        }
    }

    std::string name() const override {
        return "HybridPGMLIPP";
    }

    std::vector<std::string> variants() const override {
        return { SearchClass::name(), std::to_string(pgm_error) };
    }

    size_t size() const override {
        return dp_index_.size() + lipp_index_.size();
    }

    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& ops_filename) const override {
        return !multithread;
    }

private:
    void flush_to_lipp() {
        std::vector<KeyValue<KeyType>> snapshot;
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            snapshot.swap(insert_buffer_);
            insert_count_ = 0;
        }

        for (const auto& kv : snapshot) {
            lipp_index_.Insert(kv, 0);
        }

        // Reconstruct DPGM with empty buffer to reduce overhead
        dp_index_ = DynamicPGM<KeyType, SearchClass, pgm_error>(std::vector<int>());
        flushing_ = false;
    }

    DynamicPGM<KeyType, SearchClass, pgm_error> dp_index_;
    Lipp<KeyType> lipp_index_;

    std::vector<KeyValue<KeyType>> insert_buffer_;
    std::mutex buffer_mutex_;
    size_t insert_count_;
    size_t flush_threshold_;

    std::thread flush_thread_;
    std::atomic<bool> flushing_;
};
