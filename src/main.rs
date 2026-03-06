use simd:: conf::*;
use simd::create_tensor::*;
use simd::benchmark::*;
use faer::Mat;

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
            faer_test();
            let mut a = Mat::<f32>::zeros(100, 100);
            let mut b = Mat::<f32>::zeros(100, 100);
            a[(0, 1)] = 5.0;
            b[(0, 0)] = 10.0;
            let result = &a + &b;
            println!("{:?}", result);

            
            let a = create_tensor(10000, 10000);
            let set_a = tensor_calc(a, 10.5);
            let b = create_tensor(10000, 10000);
            let set_b = tensor_calc(b, 5.0);
            add_benchmark(set_a,set_b);
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }
}
