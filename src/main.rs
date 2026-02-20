use simd::benchmark::*;
use simd::create_tensor::*;
use simd::calc_tensor::*;
fn main() {
    println!("tensor test");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            let t =  create_tensor(3, 50);
            let result = tensor_calc(t, 179);
            println!("{:?}", result.data);
            let t1 = create_tensor(9, 5);
            let t2 = create_tensor(9, 5);
            let t1 = tensor_calc(t1, 14);
            let t2 = tensor_calc(t2, 784);
            let add_result= add_tensor(t1, t2);
            println!("{:?}", add_result.data);
            let t1 = create_tensor(10000, 10000);
            let t2 = create_tensor(10000, 10000);
            tesor_add_benchmark(t1, t2);
            foo();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }


    }
}
