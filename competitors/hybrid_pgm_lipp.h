#pragma once

#include "base.h"
#include "dynamic_pgm_index.h"
#include "lipp.h"
#include <memory>
#include <thread>
#include <atomic>
#include <mutex>
#include <vector>

template<class KeyType, class SearchClass, size_t default_pgm_error = 128>
class HybridPGMLIPP : public Competitor<KeyType, SearchClass> {
public:
    HybridPGMLIPP(const std::vector<int>& params)
        : insert_count_(0), flushing_(false) {
        flush_threshold_ = params.size() > 0 ? params[0] : 100000;
        pgm_error_ = params.size() > 1 ? params[1] : default_pgm_error;

        dp_index_ = std::make_unique<DPGMType>(std::vector<int>{});
        lipp_index_ = std::make_unique<Lipp<KeyType>>(std::vector<int>{});
    }

    ~HybridPGMLIPP() {
        if (flush_thread_.joinable()) flush_thread_.join();
    }

    uint64_t Build(const std::vector<KeyValue<KeyType>>& data, size_t num_threads) {
        return lipp_index_->Build(data, num_threads);
    }

    size_t EqualityLookup(const KeyType& key, uint32_t thread_id) const {
        size_t result = dp_index_->EqualityLookup(key, thread_id);
        return (result == util::OVERFLOW || result == util::NOT_FOUND)
            ? lipp_index_->EqualityLookup(key, thread_id)
            : result;
    }

    uint64_t RangeQuery(const KeyType& lo, const KeyType& hi, uint32_t thread_id) const {
        return dp_index_->RangeQuery(lo, hi, thread_id) + lipp_index_->RangeQuery(lo, hi, thread_id);
    }

    void Insert(const KeyValue<KeyType>& data, uint32_t thread_id) {
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            insert_buffer_.emplace_back(data);
        }

        dp_index_->Insert(data, thread_id);
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
        return { SearchClass::name(), std::to_string(pgm_error_) };
    }

    size_t size() const {
        return dp_index_->size() + lipp_index_->size();
    }

    bool applicable(bool unique, bool range_query, bool insert, bool multithread,
                    const std::string& ops_filename) const {
        return !multithread;
    }

private:
    using DPGMType = DynamicPGM<KeyType, SearchClass, default_pgm_error>;

    void flush_to_lipp() {
        std::vector<KeyValue<KeyType>> snapshot;
        {
            std::lock_guard<std::mutex> guard(buffer_mutex_);
            snapshot.swap(insert_buffer_);
            insert_count_ = 0;
        }
        for (const auto& kv : snapshot) {
            lipp_index_->Insert(kv, 0);
        }
        flushing_ = false;
    }

    std::unique_ptr<DPGMType> dp_index_;
    std::unique_ptr<Lipp<KeyType>> lipp_index_;

    std::vector<KeyValue<KeyType>> insert_buffer_;
    std::mutex buffer_mutex_;
    size_t insert_count_;
    size_t flush_threshold_;
    size_t pgm_error_;

    std::thread flush_thread_;
    std::atomic<bool> flushing_;
};
