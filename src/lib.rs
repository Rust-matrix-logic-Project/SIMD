#![feature(f16)]
use std::alloc::alloc;
pub mod benchmark;
pub mod create_tensor;
pub mod calc_tensor;
pub mod conf;
use std::ops::Add;
#[repr(align(32))]
pub struct Tensor{
pub rows: usize,
pub cols: usize,
pub data: Vec<f32>
}
impl Tensor {
    pub fn set_f16(&self,size: usize, align: usize) -> *mut f16{
        unsafe {
            let layout = std::alloc::Layout::from_size_align(size, align).unwrap();
            let ptr = alloc(layout) as *mut f16;
            if ptr.is_null() || !((ptr as usize) % 32 == 0){panic!("メモリ確保に失敗");}
            ptr
        }
        
    }
}
impl Add for Tensor {
    type Output = Tensor;
    fn add(mut self, point: Tensor) -> Tensor {
        if self.data.len() != point.data.len(){
            eprintln!("配列の要素が一致していません。");
        }   
        self.data.iter_mut().zip(point.data.iter())
        .for_each(|(a, b)| *a += b);
        self
    }
}
