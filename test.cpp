/*
compile: g++ -std=c++11 -g -O0 core_cpu.cpp test.cpp -o test.bin
*/

#define WARPRNA_ENABLE_CPU

#include <algorithm>
#include <vector>
#include <iostream>
#include <assert.h>
#include "core.h"

static bool found_any_error = false;


void assert_all_close(const float* a, const float* b, size_t size, float rtol) {
    for(size_t i = 0; i < size; ++i) {
        if(std::abs(a[i] - b[i]) > rtol * std::abs(b[i] ? b[i] : 1)) {
            std::cout << "Mismatch at index " << i << ": " << a[i] << " != " << b[i] << std::endl;
            found_any_error = true;
        }
    }
}


void test_warprna_forward() {
    float expected_costs[] = {2.6347387, 2.4651031};
    float expected_grads[] = {
       -0.34075904, -0.65924096,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        , -0.09434381,
       -0.24641524,  0.        , -0.4480959 ,  0.        , -0.2111451 ,
        0.        ,  0.        ,  0.        ,  0.        , -0.09434381,
        0.        , -0.25838017,  0.        , -0.43613094, -0.2111451 ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.35272402, -0.64727604,  0.        ,
        0.        , -0.6283351 , -0.37166485,  0.        ,  0.        ,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.26558593, -0.36274916,  0.        , -0.23790276, -0.13376209,
        0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
       -0.26558593,  0.        , -0.26772842, -0.3329236 ,  0.        ,
       -0.13376209,  0.        ,  0.        ,  0.        ,  0.        ,
        0.        ,  0.        , -0.53331435,  0.        , -0.46668565,
        0.        ,  0.
    };
    float log_probs[] = {
       -1.404937  , -0.682764  , -1.3887019 , -1.2524394 , -1.0314803 ,
       -1.0280204 , -1.1962457 , -0.9378696 , -1.1834776 , -1.0341753 ,
       -0.84465826, -1.5381583 , -0.968842  , -1.014324  , -1.355454  ,
       -0.82076967, -1.1013496 , -1.4806707 , -1.4382883 , -1.1657983 ,
       -0.7963043 , -1.3840189 , -0.8365448 , -1.1512988 , -1.0518827 ,
       -1.2960436 , -0.9752227 , -1.3433094 , -0.86678636, -1.1434443 ,
       -0.7251879 , -1.3210689 , -1.3906379 , -1.0998478 , -1.0005999 ,
       -1.2059098 , -1.0222101 , -1.4761751 , -0.88748205, -1.1836296 ,
       -0.7848896 , -1.4368956 , -1.0078475 , -1.2856646 , -1.0257446 ,
       -1.0258968 , -1.1315378 , -1.1426008 , -1.0994226 , -1.1223896 ,
       -1.0745965 , -1.0935967 , -0.89829373, -1.3558557 , -1.0778286 ,
       -0.8436196 , -1.4717846 , -1.2342478 , -1.0024878 , -1.0729997 ,
       -0.9652174 , -1.1989584 , -1.1469893 , -1.507225  , -1.1538    ,
       -0.76994103, -1.1912595 , -0.8991935 , -1.2404156 , -0.9130137 ,
       -1.1966558 , -1.2157627};
    int batch_size = 2, max_time = 4, max_u = 3, num_classes_raw = 3;
    assert(sizeof(log_probs) / sizeof(float) == batch_size * max_time * max_u * num_classes_raw);
    int labels[] = {1, 2, 1, 1};
    int input_lengths[] = {4, 4};
    int label_lengths[] = {2, 2};
    int blank_label = 0;

    int min_u = *std::min_element(label_lengths, label_lengths + batch_size); // It's actually the min(U-1) here.
    int max_s = max_time - min_u + 1;

    std::vector<float> costs(batch_size);
    std::vector<float> grads(batch_size * max_time * max_u * num_classes_raw);
    std::vector<unsigned int> counts(batch_size * max_u * 2);
    std::vector<float> alphas(batch_size * max_s * max_u);
    std::vector<float> betas(batch_size * max_s * max_u);

    run_warp_rna_cpu(
        counts.data(), alphas.data(), betas.data(), labels,
        log_probs, grads.data(), costs.data(),
        input_lengths, label_lengths,
        batch_size, max_time, max_s, max_u, num_classes_raw, blank_label);

    std::cout << "calculated costs: ";
    for(auto const& v : costs)
        std::cout << v << ' ';
    std::cout << std::endl;

    assert_all_close(costs.data(), expected_costs, batch_size, 1e-6);
    assert_all_close(grads.data(), expected_grads, batch_size * max_time * max_u * num_classes_raw, 1e-6);
}


int main() {
    test_warprna_forward();
    std::cout << (found_any_error ? "error" : "all correct") << std::endl;
    return found_any_error;
}
