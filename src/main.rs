use simd::benchmark::*;
use simd::create_tensor::*;
use simd::calc_tensor::*;
fn main() {
    println!("tensor test");
    unsafe {
        if is_x86_feature_detected!("avx2"){
            let t =  create_tensor(3, 50);
            let result = tensor_calc(t, 17.9);
            println!("{:?}", result.data);
            let t1 = create_tensor(9, 5);
            let t2 = create_tensor(9, 5);
            let t1 = tensor_calc(t1, 1.4);
            let t2 = tensor_calc(t2, 7.4);
            let add_result= add_tensor(t1, t2);
            println!("add_resulut: {:?}\n", add_result.data);
            let t1 = create_tensor(10000, 10000);
            let t2 = create_tensor(10000, 10000);
            tesor_add_benchmark(t1, t2);
            foo1();
            let t1 = create_tensor(9, 5);
            let t2 = create_tensor(9, 5);
            let t1 = tensor_calc(t1, 1.4);
            let t2 = tensor_calc(t2, 784.5);
            let mul_result= mul_tensor_elementwise(t1, t2);
            println!("mul_resulut: {:?}\n", mul_result.data);
            let t1 = create_tensor(10000, 10000);
            let t2 = create_tensor(10000, 10000);
            tesor_mul_benchmark(t1, t2);
            foo2();
            let t1 = create_tensor(9, 5);
            let t2 = create_tensor(5, 9);
            let t1 = tensor_calc(t1, 5.41);
            let t2 = tensor_calc(t2, 404.56);
            let dot_result= dot_tensor(t1, t2);
            println!("dot_resulut: {:?}\n", dot_result.data);
            let t1 = create_tensor(1000, 1000);
            let t2 = create_tensor(1000, 1000);
            tesor_dot_benchmark(t1, t2);
            foo3();
            let t1 = create_tensor(5, 7);
            let t2 = create_tensor(5, 7);
            let t1 = tensor_calc(t1, 5.41);
            let t2 = tensor_calc(t2, 404.56);
            let div_result= div_tensor(t1, t2);
            println!("div_resulut: {:?}\n", div_result.data);
            let t1 = create_tensor(1000, 1000);
            let t2 = create_tensor(1000, 1000);
            let t1 = tensor_calc(t1, 5.0);
            let t2 = tensor_calc(t2, 15.0);
            tesor_div_benchmark(t1, t2);
            foo4();
            let t =  create_tensor(3, 50);
            let result = tensor_calc(t, 17.9);
            println!("{:?}", result.data);
            let t1 = create_tensor(9, 5);
            let t2 = create_tensor(9, 5);
            let t1 = tensor_calc(t1, 9.4);
            let t2 = tensor_calc(t2, 7.4);
            let sub_result= sub_tensor(t1, t2);
            println!("sub_resulut: {:?}\n", sub_result.data);
            let t1 = create_tensor(10000, 10000);
            let t2 = create_tensor(10000, 10000);
            tesor_sub_benchmark(t1, t2);
            foo5();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }


    }
}
