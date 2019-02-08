#define _USE_MATH_DEFINES
#include <iostream>
#include <fstream>
#include <vector>
#include <optional>
#include <algorithm>
#include <random>
#include <tuple>
#include <omp.h>

struct Vec {
	double x;
	double y;
	double z;
	Vec(double v = 0)
		: Vec(v, v, v) {}
	Vec(double x, double y, double z)
		: x(x), y(y), z(z) {}
	double operator[](int i) const {
		return (&x)[i];
	}
};

Vec operator+(Vec a, Vec b) {
	return Vec(a.x + b.x, a.y + b.y, a.z + b.z);
}
Vec operator-(Vec a, Vec b) {
	return Vec(a.x - b.x, a.y - b.y, a.z - b.z);
}
Vec operator*(Vec a, Vec b) {
	return Vec(a.x * b.x, a.y * b.y, a.z * b.z);
}
Vec operator/(Vec a, Vec b) {
	return Vec(a.x / b.x, a.y / b.y, a.z / b.z);
}
Vec operator-(Vec v) {
	return Vec(-v.x, -v.y, -v.z);
}

double dot(Vec a, Vec b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}
Vec cross(Vec a, Vec b) {
	return Vec(a.y * b.z - a.z * b.y,
		a.z * b.x - a.x * b.z,
		a.x * b.y - a.y * b.x);
}
Vec normalize(Vec v) {
	return v / sqrt(dot(v, v));
}

std::tuple<Vec, Vec> tangentSpace(const Vec& n) {
	const double s = std::copysign(1, n.z);
	const double a = -1 / (s + n.z);
	const double b = n.x*n.y*a;
	return {
		Vec(1 + s * n.x*n.x*a,s*b,-s * n.x),
		Vec(b,s + n.y*n.y*a,-n.y)
	};
}

int tonemap(double v) {
	return std::min(
		std::max(int(std::pow(v, 1 / 2.2) * 255), 0), 255);
};

struct Random {
	std::mt19937 engine;
	std::uniform_real_distribution<double> dist;
	Random() {};
	Random(int seed) {
		engine.seed(seed);
		dist.reset();
	}
	double next() { return dist(engine); }
};

struct Ray {
	Vec org;
	Vec dir;
};

struct Sphere;
struct Hit {
	double t;
	Vec p;
	Vec n;
	const Sphere* sphere;
};

struct Sphere {
	Vec p;
	double r;
	Vec R;
	Vec Le;
	std::optional<Hit> intersect(
		const Ray& ray,
		double tmin,
		double tmax) const {
		const Vec op = p - ray.org;
		const double b = dot(op, ray.dir);
		const double det = b * b - dot(op, op) + r * r;
		if (det < 0) { return {}; }
		const double t1 = b - sqrt(det);
		if (tmin < t1&&t1 < tmax) { return Hit{ t1, {}, {}, this }; }
		const double t2 = b + sqrt(det);
		if (tmin < t2&&t2 < tmax) { return Hit{ t2, {}, {}, this }; }
		return {};
	}
};

struct Scene {
	std::vector<Sphere> spheres{
		{ Vec(1e5 + 1,40.8,81.6)  , 1e5 , Vec(.75,.25,.25) },
		{ Vec(-0.5,0.0,-3.0), 1e5 , Vec(.25,.25,.75) },
		{ Vec(-0.5,0.0,-3.0)      , 1e5 , Vec(.15,.75,.75) },
		{ Vec(16,43.5,16)       , 16.5, Vec(.999)  },
		{ Vec(27,16.5,47)       , 16.5, Vec(.999)  },
		{ Vec(73,16.5,78)       , 16.5, Vec(.999)  },
		{ Vec(50,681.6 - .27,81.6), 500 , Vec(), Vec(12) },
	};
	std::optional<Hit> intersect(
		const Ray& ray,
		double tmin,
		double tmax) const {
		std::optional<Hit> minh;
		for (const auto& sphere : spheres) {
			const auto h = sphere.intersect(
				ray, tmin, tmax);
			if (!h) { continue; }
			minh = h;
			tmax = minh->t;
		}
		if (minh) {
			const auto* s = minh->sphere;
			minh->p = ray.org + ray.dir * minh->t;
			minh->n = (minh->p - s->p) / s->r;
		}
		return minh;
	}
};

int main() {
	const int w = 150;
	const int h = 100;

	const int spp = 100;

	const Vec eye(50, 52, 295.6);
	const Vec center = eye + Vec(0, -0.042612, -1);
	const Vec up(0, 1, 0);
	const double fov = 30 * M_PI / 180;
	const double aspect = double(w) / h;

	const auto wE = normalize(eye - center);
	const auto uE = normalize(cross(up, wE));
	const auto vE = cross(wE, uE);

	Scene scene;
	std::vector<Vec> I(w*h);
#pragma omp parallel for schedule(dynamic, 1)
	for (int i = 0; i < w*h; i++) {
		thread_local Random rng(42 + omp_get_thread_num());
		for (int j = 0; j < spp; j++) {
			const int x = i % w;
			const int y = h - i / w;
			Ray ray;
			ray.org = eye;
			ray.dir = [&]() {
				const double tf = std::tan(fov*.5);
				const double rpx = 2.*(x + rng.next()) / w - 1;
				const double rpy = 2.*(y + rng.next()) / h - 1;
				const Vec w = normalize(
					Vec(aspect*tf*rpx, tf*rpy, -1));
				return uE * w.x + vE * w.y + wE * w.z;
			}();

			Vec L(0), th(1);
			for (int depth = 0; depth < 10; depth++) {
				const auto h = scene.intersect(
					ray, 1e-4, 1e+10);
				if (!h) {
					break;
				}
				L = L + th * h->sphere->Le;
				ray.org = h->p;
				ray.dir = [&]() {
					const auto n = dot(h->n, -ray.dir) > 0 ? h->n : -h->n;
					const auto&[u, v] = tangentSpace(n);
					const auto d = [&]() {
						const double r = sqrt(rng.next());
						const double t = 2 * M_PI*rng.next();
						const double x = r * cos(t);
						const double y = r * sin(t);
						return Vec(x, y,
							std::sqrt(
								std::max(.0, 1 - x * x - y * y)));
					}();
					return u * d.x + v * d.y + n * d.z;
				}();
				th = th * h->sphere->R;
				if (std::max({ th.x,th.y,th.z }) == 0) {
					break;
				}
			}
			I[i] = I[i] + L / spp;
		}
	}
	std::ofstream ofs("result.ppm");
	ofs << "P3\n" << w << " " << h << "\n255\n";
	for (const auto& i : I) {
		ofs << tonemap(i.x) << " "
			<< tonemap(i.y) << " "
			<< tonemap(i.z) << "\n";
	}
	return 0;
}
