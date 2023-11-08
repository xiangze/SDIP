package sdip

import chisel3._
import chisel3.util

/**
  * ALU of each of onnx operators which includes scratch pad memory genreg
  */

class SDIP_ALU extends Module{
  val io = IO(new Bundle {
    val arg1=Input(OP.W)
    val arg2=Input(OP.W)
    val weight=Input(DataType(DATA.W))
    val op=Input(UInt(OP.W))
    val dulation=Input(UInt())
    val dout=Output(UInt(DATA.W))
  })

    val conv2d=Module(new SDIP_conv2d)
    val linear=Module(new SDIP_linear)
    val transpose=Module(new SDIP_transpose)
    val shiftreg=RegInit(DataType(DATA.W))

    conv2d.io=
    
    val dout=Mux(0,Seq(
        (op_post===0 )->shiftreg
        (op_post===1 )->conv2d.io.dout
        (op_post===2 )->linear.io.dout
        (op_post===3 )->transpose.io.dout
    ))
    io.dout:=dout

}
