// competitors/hybrid_pgm_lipp.h
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
        : dp_index_(params), lipp_index_(params), insert_count_(0), flushing_(false) {
        flush_threshold_ = params.empty() ? 100000 : params[0];
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
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.emplace_back(data);
            ++insert_count_;
        }

        dp_index_.Insert(data, thread_id);

        if (insert_count_ >= flush_threshold_ && !flushing_.exchange(true)) {
            if (flush_thread_.joinable()) flush_thread_.join();  // Wait safely
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
        std::vector<KeyValue<KeyType>> buffer_snapshot;
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            buffer_snapshot.swap(insert_buffer_);
            insert_count_ = 0;
        }

        // Bulk insert into LIPP
        for (const auto& kv : buffer_snapshot) {
            lipp_index_.Insert(kv, 0);
        }

        // Rebuild dp_index_ from empty for best performance
        std::vector<KeyValue<KeyType>> empty;
        DynamicPGM<KeyType, SearchClass, pgm_error> new_dp(empty);  // Rebuild using constructor
        dp_index_ = std::move(new_dp);

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
