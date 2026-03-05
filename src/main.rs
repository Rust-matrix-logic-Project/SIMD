use simd:: conf::*;
use simd::create_tensor::*;
use simd::benchmark::*;
fn main() {
    println!("tensor test");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            test_aligment();
            let a = create_tensor(10000, 10000);
            let set_a = tensor_calc(a, 10.5);
            let b = create_tensor(10000, 10000);
            let set_b = tensor_calc(b, 10.5);

            tesor_add_benchmark(set_a, set_b);

        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }
}

