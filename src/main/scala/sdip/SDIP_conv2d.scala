package sdip

import chisel3._
import chisel3.util
class SDIP_conv2d extends Module{
  val io = IO(new Bundle {
    val in=Input(DataType(DATA.W))
    val kernel=Input(DataType(DATA.W))
    val out=Input(DataType(DATA.W))
    })
    
}