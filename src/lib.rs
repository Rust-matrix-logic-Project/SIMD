pub mod benchmark;
pub mod create_tensor;
pub mod calc_tensor;

#[repr(align(32))]
pub struct Tensor{
   pub  rows: usize,
   pub cols: usize,
   pub data: Vec<i32>
}