#pragma once

#include <memory>

class CarFlare {
public:
    CarFlare();
    ~CarFlare();

    CarFlare(const CarFlare&) = delete;
    CarFlare& operator=(const CarFlare&) = delete;
    CarFlare(CarFlare&&) noexcept;
    CarFlare& operator=(CarFlare&&) noexcept;

    void run();

private:
    class Impl;
    std::unique_ptr<Impl> impl;
};
