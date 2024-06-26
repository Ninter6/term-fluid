#include <iostream>
#include <vector>
#include <numeric>
#include <thread>
#include <unistd.h>
#include <random>

#include "mathpls.h"

using namespace mathpls;

constexpr size_t wth = 80, hgt = 20;
constexpr float s = 2.f; // Smooth radius

char img[wth * hgt + hgt]{};

struct Particle {
    Particle(const vec2& p) : pos(p) {}
    vec2 pos{};
    vec2 vel{};
    vec2 acc{};
    float density;
}; // The mass of each particle is 1
std::vector<Particle> p;
std::vector<int> gird[wth+1][hgt+1]{};
std::vector<int> near_gird[wth+1][hgt+1]{};

std::vector<ivec2> bar; // barriers

float target_density = .8f;
float K = 100.f; // The pressure multiplier
float V = 2.f; // The viscosity strength
float E = 1.f; // The surface tension coefficient

constexpr float R = 1.f, D = .1f; // edge constant

vec2 g = {0, 1.f};

std::vector<int>& get_near_particles(int n);

// kernel
float smooth(float r) {
    if (r >= s) return 0;
    constexpr float v = 10.f/(M_PI*s*s*s*s*s);
    return (s-r)*(s-r)*(s-r)*v;
}
float grad_smooth(float r) {
    if (r >= s) return 0;
    constexpr float v = 6.f/(s*s*s*s*M_PI);
    return -(s-r)*(s-r)*v;
}
float grad2_smooth(float r) {
    if (r >= s) return 0;
    constexpr float v = 3.f/(s*s*s*M_PI);
    return (s-r)*v;
}

float viscosity_kernel(float r) {
    if (r >= s) return 0;
    constexpr float v = 1.f/(M_PI_4*s*s*s*s*s*s*s*s);
    float a = s*s - r*r;
    return a*a*a*v;
}

float density(int n) {
    const auto& np = get_near_particles(n);
    return std::transform_reduce(np.begin(), np.end(), 0.f,
                                 std::plus<float>{}, [n](int i) {
        return smooth(distance(p[n].pos, p[i].pos));
    });
}

float density2pressure(float density) {
    float de = density - target_density;
    float pressure = de * K;
    return max(pressure, 0);
}

vec2 surface_tension(int n) {
    const auto& np = get_near_particles(n);
    vec2 c1{};
    float c2{};
    for (auto&& i : np) {
        if (i == n) continue;
        auto d = p[i].pos - p[n].pos;
        auto l = d.length();
        if (l == 0) d = mathpls::random::rand_vec2();
        else d /= l;
        c1 += d * grad_smooth(l) / p[i].density;
        c2 += grad2_smooth(l) / p[i].density;
    }
    return -E * c2 * c1.normalized();
}

vec2 pressure_force(int n) {
    auto P = density2pressure(p[n].density);
    const auto& np = get_near_particles(n);
    return std::transform_reduce(np.begin(), np.end(), vec2{},
                                 std::plus<vec2>{}, [=](int i) {
        if (i == n) return vec2{};
        auto d = p[i].pos - p[n].pos;
        auto l = d.length();
        if (l == 0) d = mathpls::random::rand_vec2();
        else d /= l;
        return d * grad_smooth(l) * (density2pressure(p[n].density) + P) * .5f / p[i].density;
    });
}

vec2 viscosity_force(int n) {
    const auto& np = get_near_particles(n);
    return std::transform_reduce(np.begin(), np.end(), vec2{},
                                 std::plus<vec2>{}, [n](int i) {
        return (p[i].vel - p[n].vel) * viscosity_kernel(distance(p[n].pos, p[i].pos));
    }) * V;
}

void calcu_density() {
    for (int i = 0; i < p.size(); i++)
        p[i].density = density(i);
}

void calcu_force() {
    for (int i = 0; i < p.size(); i++) {
        auto force = pressure_force(i) + viscosity_force(i)/* + surface_tension(i)*/;
        p[i].acc = force / p[i].density + g;
    }
}

void edge_detect(int i) {
    if (p[i].pos.x < 0 && p[i].vel.x < 0)
        p[i].vel.x *= -R;
    if (p[i].pos.y < 0 && p[i].vel.y < 0)
        p[i].vel.y *= -R;
    if (p[i].pos.x > wth && p[i].vel.x > 0)
        p[i].vel.x *= -R;
    if (p[i].pos.y > hgt && p[i].vel.y > 0)
        p[i].vel.y *= -R;

    p[i].pos.x = std::clamp<float>(p[i].pos.x, 0, wth);
    p[i].pos.y = std::clamp<float>(p[i].pos.y, 0, hgt);
}

void solve_barrier() {
    for (auto&& b : bar) {
        auto [x, y] = b.asArray;
        if (y > 0) { // up
            auto& a = gird[x][y-1];
            for (auto&& i : a)
                if (y - p[i].pos.y < D + .5f && p[i].vel.y > 0) {
                    p[i].pos.y = y - D - .5f;
                    p[i].vel.y *= -R;
                }
        }
        if (y < hgt) { // down
            auto& a = gird[x][y+1];
            for (auto&& i : a)
                if (p[i].pos.y - y < D + .5f && p[i].vel.y < 0) {
                    p[i].pos.y = y + D + .5f;
                    p[i].vel.y *= -R;
                }
        }
        if (x > 0) { // left
            auto& a = gird[x-1][y];
            for (auto&& i : a)
                if (x - p[i].pos.x < D + .5f && p[i].vel.x > 0) {
                    p[i].pos.x = x - D - .5f;
                    p[i].vel.x *= -R;
                }
        }
        if (x < wth) { // right
            auto& a = gird[x+1][y];
            for (auto&& i : a)
                if (p[i].pos.x - x < D + .5f && p[i].vel.x < 0) {
                    p[i].pos.x = x + D + .5f;
                    p[i].vel.x *= -R;
                }
        }
    }
}

void calcu_movement(float dt) {
    for (int i = 0; i < p.size(); i++) {
        p[i].vel += p[i].acc * dt;
        p[i].pos += p[i].vel * dt;

        edge_detect(i);
    }
}

void init() {
    random::seed(std::random_device{}());

    for (int i = 0; i < hgt-1; i++)
        img[wth*i+i+wth] = '\n';

    std::string buf, tb = "1234567890-=+*@█", bt = "#";
    for (int i = 0; std::getline(std::cin, buf) && i <= hgt; i++)
        for (int j = 0; j < min(wth+1, buf.size()); j++)
            if (tb.find(buf[j]) < tb.size()) {
                for (int n = 0; n < 5; n++)
                    p.emplace_back(vec2(j, i) + mathpls::random::rand_vec2() * 0.1f);
            } else if (bt.find(buf[j]) < bt.size()) {
                bar.emplace_back(j, i);
            }
}

ivec2 part_gird_pos(int n) {
    return {(int)round(p[n].pos.x), (int)round(p[n].pos.y)};
}

void fill_near_grid() {
    for (int x = 0; x <= wth; x++)
        for (int y = 0; y <= hgt; y++) {
            if (gird[x][y].empty()) continue;
            near_gird[x][y].clear();
            for (int i = x - s; i <= x + s; i++)
                for (int j = y - s; j <= y + s; j++)
                    if (i >= 0 && i <= wth && j >= 0 && j <= hgt)
                        near_gird[x][y].insert(near_gird[x][y].end(), gird[i][j].begin(), gird[i][j].end());
        }
}

void fill_gird() {
    for (int i = 0; i < p.size(); i++) {
        auto [x, y] = part_gird_pos(i).asArray;
        gird[x][y].push_back(i);
    }
    fill_near_grid();
}

std::vector<int>& get_near_particles(int n) {
    auto [x, y] = part_gird_pos(n).asArray;
    return near_gird[x][y];
}

void clr_gird() {
    for (int i = 0; i <= wth; i++)
        for (int j = 0; j <= hgt; j++)
            gird[i][j].clear();
}

uint8_t rd_smp(int x, int y) {
    if (!gird[x][y].empty()) return true;
    auto f = std::lower_bound(bar.begin(), bar.end(), ivec2{x, y}, [](auto&& a, auto&& b){
        return a.y < b.y || (a.y == b.y && a.x < b.x);
    });
    return f != bar.end() && f->x == x && f->y == y;
}

void gen_img() {
    constexpr auto lut = R"( .,_`/|''|\`^,.#)";
    for (int i = 0; i < wth; i++)
        for (int j = 0; j < hgt; j++){
            uint8_t c =
                    (rd_smp(i,   j)   << 3) |
                    (rd_smp(i+1, j)   << 2) |
                    (rd_smp(i+1, j+1) << 1) |
                    (rd_smp(i,   j+1));
            img[wth*j+j+i] = lut[c];
        }
}

void print() {
    std::puts(img);
}
void clr_scr() {
    std::puts("\033[2J\033[H");
}

void step(float dt) {
    clr_gird();
    fill_gird();
    calcu_density();
    calcu_force();
    calcu_movement(dt);
    solve_barrier();
}

void idem() {
    fill_gird();
    gen_img();
    clr_scr();
    print();
    puts("Just a second...");
    sleep(1);
}

int main() {
    init();
    idem();

    while (1) {
        step(1.f/256.f);
        gen_img();
        clr_scr();
        print();
    }

    return 0;
}
