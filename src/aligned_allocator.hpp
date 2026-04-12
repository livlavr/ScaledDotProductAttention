#pragma once

#include <cstddef>
#include <memory>

namespace attention {

    template <typename T, std::size_t Alignment>
    struct AlignedAllocator {
        using value_type = T;

        AlignedAllocator() = default;

        template <typename U>
        explicit AlignedAllocator(const AlignedAllocator<U, Alignment>&) noexcept {};

        T* allocate(const std::size_t count) {
            if (count == 0) {
                return nullptr;
            }

            std::size_t size = count * sizeof(T);
            if (size % Alignment != 0) {
                size += (Alignment - size % Alignment);
            }

            void* ptr = std::aligned_alloc(Alignment, size);
            if (not ptr) {
                throw std::bad_alloc();
            }

            return static_cast<T*>(ptr);
        }

        void deallocate(T* p, std::size_t) noexcept {
            std::free(p);
        }

        template <typename U>
        struct rebind {
            using other = AlignedAllocator<U, Alignment>;
        };

        template <typename U>
        bool operator==(const AlignedAllocator<U, Alignment>&) const noexcept {
            return true;
        }
    };

} // namespace attention