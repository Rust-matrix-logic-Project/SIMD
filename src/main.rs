use simd::benchmark::*;
fn main() {
    println!("test start");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            foo();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }
}
