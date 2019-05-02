#include <array>
#include <cmath>
#include <random>
#include <thread>
#include <unordered_map>
#include <algorithm>
#include <iostream>

#include <src/tools.hpp>

template <uint16_t Dim>
class Point {
public:
    inline float& operator[](size_t i) {
        return data_[i];
    }

    inline float operator[](size_t i) const {
        return data_[i];
    }

    float Norm() const {
        float max = 0;
        for (auto x : data_) {
            float y = fabsf(x);
            if (y > max) {
                max = y;
            }
        }

        return max;
    }

    float Dist(Point other) const {
        other -= *this;
        return other.Norm();
    }

    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    Point& operator/=(T val) {
        for (auto& x : data_) {
            x /= val;
        }
        return *this;
    }

    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    Point operator/(T val) const {
        Point ret(*this);
        ret /= val;
        return ret;
    }

    Point& operator+=(const Point& other) {
        for (int i = 0; i < Dim; ++i) {
            data_[i] += other[i];
        }
        return *this;
    }

    Point& operator-=(const Point& other) {
        for (int i = 0; i < Dim; ++i) {
            data_[i] -= other[i];
        }
        return *this;
    }

    Point operator-(const Point& other) const {
        auto ret = *this;
        ret -= other;
        return ret;
    }

private:
    std::array<float, Dim> data_;
};

template <uint16_t Dim>
static Point<Dim> ZeroPoint() {
    Point<Dim> point;
    for (int i = 0; i < Dim; ++i) {
        point[i] = 0;
    }
    return point;
}

template <uint16_t Dim, class RandomDeviceT>
static Point<Dim> RandomPoint(RandomDeviceT* rd) {
    std::uniform_real_distribution<float> distr(-1, 1);
    Point<Dim> point;
    while (true) {
        for (int i = 0; i < Dim; ++i) {
            point[i] = distr(*rd);
        }
        if (point.Norm() <= 1) {
            break;
        }
    }
    return point;
}

template <uint16_t Dim, class RandomDeviceT>
class Voronoi {
public:
    Voronoi(size_t num_points, RandomDeviceT* rd) : points_(num_points), rd_(rd) {
        for (auto& point : points_) {
            point = RandomPoint<Dim>(rd);
        }
    }

    size_t FindNNIndex(const Point<Dim>& point) const {
        size_t min_index = 0;
        float min_dist = std::numeric_limits<float>::max();
        for (size_t i = 0; i < points_.size(); ++i) {
            float dist = points_[i].Dist(point);
            if (dist < min_dist) {
                min_dist = dist;
                min_index = i;
            }
        }
        return min_index;
    }

private:
    std::vector<Point<Dim>> points_;
    RandomDeviceT* rd_;
};

template <uint16_t Dim>
using ClusterT = std::vector<Point<Dim>>;

template <uint16_t Dim, class RandomDeviceT>
static std::vector<ClusterT<Dim>> GetPointClusters(size_t voronoi_size, size_t point_cloud_size) {
    RandomDeviceT rd(rand());
    Voronoi<Dim, RandomDeviceT> voronoi(voronoi_size, &rd);
    std::vector<std::thread> workers;
    using worker_result_t = std::vector<std::pair<Point<Dim>, size_t>>;

    int num_workers = std::thread::hardware_concurrency();
    std::vector<worker_result_t> results(num_workers);
    for (int i = 0; i < num_workers; ++i) {
        workers.emplace_back(
            [& my_res_part = results[i], work_amount = point_cloud_size / num_workers, &voronoi] {
                my_res_part.reserve(256);
                RandomDeviceT rd(rand());
                for (size_t i = 0; i < work_amount; ++i) {
                    my_res_part.emplace_back(RandomPoint<Dim>(&rd), 0);
                    size_t& closest = my_res_part.back().second;
                    closest = voronoi.FindNNIndex(my_res_part.back().first);
                }
            });
    }

    for (auto& worker : workers) {
        worker.join();
    }

    std::vector<ClusterT<Dim>> work_result(voronoi_size);
    for (const auto& worker_result : results) {
        for (const auto& [point, nn] : worker_result) {
            work_result[nn].push_back(point);
        }
    }

    return work_result;
}

template <uint16_t Dim, class Func>
static auto ApplyFuncToClusters(const std::vector<ClusterT<Dim>>& clusters, Func function) {
    std::vector<decltype(function(std::declval<ClusterT<Dim>>()))> result(clusters.size());
    std::vector<std::thread> workers;
    size_t num_workers = std::thread::hardware_concurrency();
    {
        size_t step = clusters.size() / num_workers;
        for (size_t i = 0; i < num_workers; ++i) {
            size_t begin = i * step;
            size_t end = i + 1 == num_workers ? clusters.size() : (i + 1) * step;
            workers.emplace_back([begin, end, &clusters, &result, function] {
                for (size_t cluster_ix = begin; cluster_ix < end; ++cluster_ix) {
                    result[cluster_ix] = function(clusters[cluster_ix]);
                }
            });
        }
    }

    for (auto& worker : workers) {
        worker.join();
    }

    return result;
}

int main() {
    static const uint16_t Dim = 8;
    srand(1234);

    /*
    auto calc_diam = [](const auto& cluster) {
        float diam = 0;
        for (size_t i = 0; i < cluster.size(); ++i) {
            for (size_t j = 0; j < i; ++j) {
                float test = cluster[i].Dist(cluster[j]);
                if (test > diam) {
                    diam = test;
                }
            }
        }
        return diam;
    };
    */

    auto calc_stddev = [](const auto& cluster) {
        auto center = ZeroPoint<Dim>();
        size_t size = cluster.size();
        for (const auto& point : cluster) {
            center += point / size;
        }

        float stddev = 0;
        for (const auto& point : cluster) {
            float dist = point.Dist(center);
            stddev += (dist * dist) / size;
        }

        return sqrtf(stddev);
    };

    auto calc_median = [](const auto& cluster) {
        if (cluster.empty()) {
            return float(0);
        }
        auto center = ZeroPoint<Dim>();
        size_t size = cluster.size();
        for (const auto& point : cluster) {
            center += point / size;
        }

        std::vector<float> dists;
        dists.reserve(cluster.size());
        for (const auto& point : cluster) {
            dists.push_back(point.Dist(center));
        }

        std::nth_element(dists.begin(), dists.begin() + dists.size() / 2, dists.end());
        return dists[dists.size() / 2];
    };

    {
        auto clusters = GetPointClusters<Dim, std::mt19937>(128, 1 << 26);

        std::vector<float> stddevs = ApplyFuncToClusters(clusters, calc_stddev);
        std::sort(stddevs.begin(), stddevs.end());
        std::cout << "Stddevs from center:\n" << stddevs << "\n\n";

        std::vector<float> medians = ApplyFuncToClusters(clusters, calc_median);
        std::sort(medians.begin(), medians.end());
        std::cout << "Medians from center:\n" << medians << "\n\n";
    }
    return 0;
}
