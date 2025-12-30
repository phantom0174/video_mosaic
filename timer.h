#include <chrono>


struct Timer {
    using clock = std::chrono::steady_clock;
    using time_point = clock::time_point;
    using duration = clock::duration;

    time_point start_{};
    duration accum_{duration::zero()};
    bool running_{false};

    void start() { start_ = clock::now(); }
    void stop() { accum_ += clock::now() - start_; }
    void reset() { accum_ = duration::zero(); start(); }

    template<class D = std::chrono::duration<double>>
    D elapsed() const {
        if (running_) {
            return std::chrono::duration_cast<D>(accum_ + (clock::now() - start_));
        } else {
            return std::chrono::duration_cast<D>(accum_);
        }
    }

    long long milliseconds() const {
        return elapsed<std::chrono::milliseconds>().count();
    }
    long long microseconds() const {
        return elapsed<std::chrono::microseconds>().count();
    }
};
