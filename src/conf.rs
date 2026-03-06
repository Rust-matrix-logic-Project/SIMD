use crate::{Tensor, create_tensor::*};
use std::fs::File;
use std::io::Write;

pub unsafe fn test_conf(){
    unsafe {
        let test_a = create_tensor(1000, 1000);
        let test_a = tensor_calc(test_a, 10.5);
        let mut file = File::create("address/test.txt").expect("書き込みに失敗");
        for i in 0..test_a.data.len(){
            let index = test_a.data.as_ptr().add(i);
            let _ = writeln!(file, "{:?}", index);
        }
    }
}
pub unsafe fn test_align(){
    println!("アライメント確認");
    let test_align = std::mem::align_of::<Tensor>();
    println!("{}", test_align);
}

pub unsafe fn test_aligment(){
    println!("手動配置学習");
    unsafe{
        let test_align = create_tensor(100, 100).set_f16(10, 32);
        println!("{:?}", test_align);
        let a = test_align.add(1);
        println!("{:?}", a);
        test_align.write(1.5);
        println!("{}",test_align.read());
    }
    
}