use simd::conf::*;
//use simd::calc_tensor::*;
fn main() {
    println!("tensor test");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            test_align();
            test_conf();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }
}

