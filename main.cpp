#include "opencl.hpp"
#include "utilities.hpp"
#include <random>

const uint N = 10000000u; // size of vectors
float refA[N], refB[N];
float A[N], B[N], C[N];

void classicalAdd() {

	for (uint n = 0u; n < N; n++) {
		A[n] = refA[n];
		B[n] = refB[n];
	}

	Clock clock;

	for (int i = 0; i < N; i++) {
		C[i] = A[i] + B[i];
	}

	std::cout << "classicalAdd runtime: " << clock.stop() << std::endl;
}

void parallelAdd() {
	Device device(select_device_with_most_flops());

	Memory<float> A(device, N);
	Memory<float> B(device, N);
	Memory<float> C(device, N);

	for (uint n = 0u; n < N; n++) {
		A[n] = refA[n];
		B[n] = refB[n];
	}

	Kernel add_kernel(device, N, "add_kernel", A, B, C);

	Clock clock;

	A.write_to_device();
	B.write_to_device();
	add_kernel.run();
	C.read_from_device();

	std::cout << "parallelAdd runtime: " << clock.stop() << std::endl;
}

int main() {
	std::random_device dev;
	std::mt19937 rng(dev());
	std::uniform_real_distribution<float> dist{0.f, 1.f};
	
	for (uint n = 0u; n < N; n++) {
		refA[n] = dist(rng);
		refB[n] = dist(rng);
	}

	classicalAdd();
	parallelAdd();
	
	return 0;
}