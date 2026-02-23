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

            /*ブロードキャスト */
            let tensor1 =  create_tensor(10, 4);
            let tensor2 =  create_tensor(10, 4);
            let tensor1 = tensor_calc(tensor1, 5.6);
            let tensor2 = tensor_calc(tensor2, 5.6);
            
            let add_result = add_tensor(tensor1, tensor2);
            let tensor3 = create_tensor(50, 4);
            let tensor3 = tensor_calc(tensor3, 7.9);
            let result = add_cast(add_result, tensor3);
            println!("加算ブロードキャスト:\n{:?}\n",result.data);
            foo6();
            let tensor1 =  create_tensor(10, 4);
            let tensor2 =  create_tensor(10, 4);
            let tensor1 = tensor_calc(tensor1, 5.0);
            let tensor2 = tensor_calc(tensor2, 20.0);
            let mul_result = mul_tensor_elementwise(tensor1, tensor2);
            let tensor3 = create_tensor(5, 4);
            let tensor3 = tensor_calc(tensor3, 10.0);
            let result = mul_cast(mul_result, tensor3);
            println!("乗算ブロードキャスト:\n{:?}\n",result.data);
            foo7();
            let tensor1 =  create_tensor(10, 4);
            let tensor2 =  create_tensor(10, 4);
            let tensor1 = tensor_calc(tensor1, 20.6);
            let tensor2 = tensor_calc(tensor2, 10.6);
            
            let sub_result = sub_tensor(tensor1, tensor2);
            let tensor3 = create_tensor(5, 4);
            let tensor3 = tensor_calc(tensor3, 9.9);
            let result = sub_cast(sub_result, tensor3);
            println!("減算ブロードキャスト:\n{:?}\n",result.data);
            foo8();

            let tensor1 =  create_tensor(10, 4);
            let tensor2 =  create_tensor(10, 4);
            let tensor1 = tensor_calc(tensor1, 20.0);
            let tensor2 = tensor_calc(tensor2, 5.0);
            let div_result = div_tensor(tensor1, tensor2);
            let tensor3 = create_tensor(5, 4);
            let tensor3 = tensor_calc(tensor3, 2.0);
            let result = div_cast(div_result, tensor3);
            println!("除算ブロードキャスト:\n{:?}\n",result.data);
            foo9();
        }else {
            eprintln!("avx2がCPUによってサポートされていません。");
            return;           
        }


    }
}

