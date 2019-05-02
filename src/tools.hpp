#pragma once

#include <iostream>
#include <vector>

template <class T>
std::ostream& operator<<(std::ostream& stream, const std::vector<T>& vec) {
    stream << "{";
    for (auto it = vec.begin(); it != vec.end(); ++it) {
        stream << *it;
        if (std::next(it) != vec.end()) {
            stream << ", ";
        }
    }
    stream << "}";
    return stream;
}
