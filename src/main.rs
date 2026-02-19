//use simd::benchmark::*;
use simd::create_tensor::*;
fn main() {
    /*println!("test start");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            foo();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }

    }*/

    println!("tensor test");
    unsafe {
         
            let t =  create_tensor(10, 5);
            
            let result = tensor_calc(t, 10);
            println!("{:?}", result.data);

    }
}
