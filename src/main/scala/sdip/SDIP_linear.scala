package sdip

import chisel3._
import chisel3.util
class SDIP_linear extends Module{
val io = IO(new Bundle {
    val A=Input(DataType(DATA.W))
    val x=Input(DataType(DATA.W))
    val out=Input(DataType(DATA.W))
    })
  
}
