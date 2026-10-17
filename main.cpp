#include "carflare.hpp"

#include <exception>
#include <iostream>

int main() {
    try {
        CarFlare app;
        app.run();
    } catch (const std::exception& e) {
        std::cerr << "Car Flare error: " << e.what() << '\n';
        return 1;
    }

    return 0;
}
