#include <SFML/Graphics.hpp>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <omp.h>

// g++ -o nbody nbody.cpp -lsfml-graphics -lsfml-window -lsfml-system -fopenmp

const int N = 2000;
const double dt = 0.01;
const double G = 6;
const double radiusCoefficient = 0.2;

struct Particle {
    double x, y;
    double vx, vy;
    double mass;
    double ax, ay;
    double old_ax, old_ay;
};

std::vector<Particle> particles(N);
std::vector<sf::CircleShape> shapes(N);

void computeForces() {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        particles[i].old_ax = particles[i].ax;
        particles[i].old_ay = particles[i].ay;

        particles[i].ax = 0;
        particles[i].ay = 0;

        for (int j = 0; j < N; ++j) {
            if (i != j) {
                double dx = particles[j].x - particles[i].x;
                double dy = particles[j].y - particles[i].y;
                double distance = std::sqrt(dx * dx + dy * dy);
                if (distance > 0) {
                    double force = G * particles[i].mass * particles[j].mass / (distance * distance);
                    particles[i].ax += force * dx / (distance * particles[i].mass);
                    particles[i].ay += force * dy / (distance * particles[i].mass);
                }
            }
        }
    }
}

void initializeParticles() {
    for (int i = 0; i < N; ++i) {
        particles[i].mass = static_cast<double>(rand()) / RAND_MAX * 100 + 50; 
        particles[i].x = static_cast<double>(rand()) / RAND_MAX * 800;
        particles[i].y = static_cast<double>(rand()) / RAND_MAX * 600;
        particles[i].vx = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 500;
        particles[i].vy = (static_cast<double>(rand()) / RAND_MAX - 0.5) * 500;

        shapes[i].setRadius(radiusCoefficient * std::sqrt(particles[i].mass)); 
        shapes[i].setFillColor(sf::Color::White);
        shapes[i].setOrigin(shapes[i].getRadius(), shapes[i].getRadius());
    }
}

void updatePositionsAndVelocities() {
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        particles[i].x += particles[i].vx * dt + 0.5 * particles[i].ax * dt * dt;
        particles[i].y += particles[i].vy * dt + 0.5 * particles[i].ay * dt * dt;

        particles[i].vx += 0.5 * (particles[i].old_ax + particles[i].ax) * dt;
        particles[i].vy += 0.5 * (particles[i].old_ay + particles[i].ay) * dt;

        shapes[i].setPosition(particles[i].x, particles[i].y);
    }
}

void drawParticles(sf::RenderWindow& window) {
    window.clear();
    
    for (const auto& shape : shapes) {
        window.draw(shape);
    }

    window.display();
}

int main(int argc, char** argv) {
    int numThreads;

    if (argc > 1) {
        numThreads = std::atoi(argv[1]);
    } else {
        numThreads = omp_get_max_threads();
    }

    omp_set_num_threads(numThreads);
    sf::RenderWindow window(sf::VideoMode(800, 600), "N-body Simulation");
    initializeParticles();

    while (window.isOpen()) {
        sf::Event event;
        while (window.pollEvent(event)) {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        computeForces();
        updatePositionsAndVelocities();
        drawParticles(window);
    }

    return 0;
}