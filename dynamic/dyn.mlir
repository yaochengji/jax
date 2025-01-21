module @jit_call_exported attributes {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} {
  func.func public @main(%arg0: tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, %arg1: tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>> {jax.result_info = "[0]"}) {
    %2 = call @call_exported_matmul(%arg0, %arg1) : (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>
    return %2 : tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>
  }
  func.func private @call_exported_matmul(%arg0: tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, %arg1: tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>> {jax.result_info = ""}) {
    %c = stablehlo.constant dense<0> : tensor<i32>
    %c_0 = stablehlo.constant dense<-1> : tensor<i32>
    %c_1 = stablehlo.constant dense<128> : tensor<i32>
    %c_2 = stablehlo.constant dense<1> : tensor<i32>
    %0 = stablehlo.get_dimension_size %arg0, dim = 0 : (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> tensor<i32>
    %1 = stablehlo.get_dimension_size %arg0, dim = 1 : (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> tensor<i32>
    %2 = stablehlo.get_dimension_size %arg1, dim = 0 : (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> tensor<i32>
    %3 = stablehlo.get_dimension_size %arg1, dim = 1 : (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> tensor<i32>
    %4 = stablehlo.compare  GE, %0, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%4, %0) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'm'. Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {0} from specification 'm' for dimension args[0].shape[0] (= {0}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>) -> ()
    %5 = stablehlo.compare  GE, %1, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%5, %0, %1) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'l'. Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {0} from specification 'm' for dimension args[0].shape[0] (= {0}), 'l' = {1} from specification 'l' for dimension args[0].shape[1] (= {1}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>) -> ()
    %6 = stablehlo.compare  EQ, %2, %1,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%6, %2, %1, %0) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Found inconsistency between dimension size args[1].shape[0] (= {0}) and the specification 'l' (= {1}). Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {2} from specification 'm' for dimension args[0].shape[0] (= {2}), 'l' = {1} from specification 'l' for dimension args[0].shape[1] (= {1}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %7 = stablehlo.compare  GE, %3, %c_2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    stablehlo.custom_call @shape_assertion(%7, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the polymorphic shapes specification. Expected value >= 1 for dimension variable 'n'. Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {0} from specification 'm' for dimension args[0].shape[0] (= {0}), 'l' = {1} from specification 'l' for dimension args[0].shape[1] (= {1}), 'n' = {2} from specification 'n' for dimension args[1].shape[1] (= {2}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %8:2 = call @divmod_0(%0, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %9 = stablehlo.multiply %8#0, %c_1 : tensor<i32>
    %10 = stablehlo.convert %0 : tensor<i32>
    %11 = stablehlo.multiply %10, %c_0 : tensor<i32>
    %12 = stablehlo.add %9, %11 : tensor<i32>
    %13 = stablehlo.compare  EQ, %12, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %14:2 = call @divmod_0(%0, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %15 = stablehlo.multiply %14#0, %c_1 : tensor<i32>
    %16 = stablehlo.convert %0 : tensor<i32>
    %17 = stablehlo.multiply %16, %c_0 : tensor<i32>
    %18 = stablehlo.add %15, %17 : tensor<i32>
    stablehlo.custom_call @shape_assertion(%13, %18, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint 128*floordiv(m, 128) == m. Expected '128*floordiv(m, 128) - m' to be equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {1} from specification 'm' for dimension args[0].shape[0] (= {1}), 'l' = {2} from specification 'l' for dimension args[0].shape[1] (= {2}), 'n' = {3} from specification 'n' for dimension args[1].shape[1] (= {3}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %19:2 = call @divmod_0(%0, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %20 = stablehlo.compare  EQ, %19#1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %21:2 = call @divmod_0(%0, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    stablehlo.custom_call @shape_assertion(%20, %21#1, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint mod(m, 128) == 0. Expected 'mod(m, 128)' to be equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {1} from specification 'm' for dimension args[0].shape[0] (= {1}), 'l' = {2} from specification 'l' for dimension args[0].shape[1] (= {2}), 'n' = {3} from specification 'n' for dimension args[1].shape[1] (= {3}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %22:2 = call @divmod_0(%3, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %23 = stablehlo.multiply %22#0, %c_1 : tensor<i32>
    %24 = stablehlo.convert %3 : tensor<i32>
    %25 = stablehlo.multiply %24, %c_0 : tensor<i32>
    %26 = stablehlo.add %23, %25 : tensor<i32>
    %27 = stablehlo.compare  EQ, %26, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %28:2 = call @divmod_0(%3, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %29 = stablehlo.multiply %28#0, %c_1 : tensor<i32>
    %30 = stablehlo.convert %3 : tensor<i32>
    %31 = stablehlo.multiply %30, %c_0 : tensor<i32>
    %32 = stablehlo.add %29, %31 : tensor<i32>
    stablehlo.custom_call @shape_assertion(%27, %32, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint 128*floordiv(n, 128) == n. Expected '128*floordiv(n, 128) - n' to be equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {1} from specification 'm' for dimension args[0].shape[0] (= {1}), 'l' = {2} from specification 'l' for dimension args[0].shape[1] (= {2}), 'n' = {3} from specification 'n' for dimension args[1].shape[1] (= {3}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %33:2 = call @divmod_0(%3, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %34 = stablehlo.compare  EQ, %33#1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %35:2 = call @divmod_0(%3, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    stablehlo.custom_call @shape_assertion(%34, %35#1, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint mod(n, 128) == 0. Expected 'mod(n, 128)' to be equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {1} from specification 'm' for dimension args[0].shape[0] (= {1}), 'l' = {2} from specification 'l' for dimension args[0].shape[1] (= {2}), 'n' = {3} from specification 'n' for dimension args[1].shape[1] (= {3}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %36:2 = call @divmod_0(%1, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %37 = stablehlo.multiply %36#0, %c_1 : tensor<i32>
    %38 = stablehlo.convert %1 : tensor<i32>
    %39 = stablehlo.multiply %38, %c_0 : tensor<i32>
    %40 = stablehlo.add %37, %39 : tensor<i32>
    %41 = stablehlo.compare  EQ, %40, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %42:2 = call @divmod_0(%1, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %43 = stablehlo.multiply %42#0, %c_1 : tensor<i32>
    %44 = stablehlo.convert %1 : tensor<i32>
    %45 = stablehlo.multiply %44, %c_0 : tensor<i32>
    %46 = stablehlo.add %43, %45 : tensor<i32>
    stablehlo.custom_call @shape_assertion(%41, %46, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint 128*floordiv(l, 128) == l. Expected '128*floordiv(l, 128) - l' to be equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {1} from specification 'm' for dimension args[0].shape[0] (= {1}), 'l' = {2} from specification 'l' for dimension args[0].shape[1] (= {2}), 'n' = {3} from specification 'n' for dimension args[1].shape[1] (= {3}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %47:2 = call @divmod_0(%1, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %48 = stablehlo.compare  EQ, %47#1, %c,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %49:2 = call @divmod_0(%1, %c_1) : (tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    stablehlo.custom_call @shape_assertion(%48, %49#1, %0, %1, %3) {api_version = 2 : i32, error_message = "Input shapes do not match the symbolic shape constraint mod(l, 128) == 0. Expected 'mod(l, 128)' to be equal to 0, but found {0}.  Using the following polymorphic shapes specifications: args[0].shape = (m, l),args[1].shape = (l, n). Obtained dimension variables: 'm' = {1} from specification 'm' for dimension args[0].shape[0] (= {1}), 'l' = {2} from specification 'l' for dimension args[0].shape[1] (= {2}), 'n' = {3} from specification 'n' for dimension args[1].shape[1] (= {3}), . Please see https://jax.readthedocs.io/en/latest/export/shape_poly.html#shape-assertion-errors for more details.", has_side_effect = true} : (tensor<i1>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> ()
    %50 = call @_wrapped_jax_export_main(%1, %0, %3, %arg0, %arg1) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>
    return %50 : tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>
  }
  func.func private @_wrapped_jax_export_main(%arg0: tensor<i32> {jax.global_constant = "l"}, %arg1: tensor<i32> {jax.global_constant = "m"}, %arg2: tensor<i32> {jax.global_constant = "n"}, %arg3: tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, %arg4: tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>) -> (tensor<?x?xf32, #stablehlo.bounds<2048, 2048>> {jax.result_info = ""}) {
    %c = stablehlo.constant dense<128> : tensor<i32>
    %0:2 = call @divmod(%arg0, %arg1, %arg2, %arg1, %c) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %1:2 = call @divmod(%arg0, %arg1, %arg2, %arg2, %c) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %2:2 = call @divmod(%arg0, %arg1, %arg2, %arg0, %c) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> (tensor<i32>, tensor<i32>)
    %3 = stablehlo.reshape %arg1 : (tensor<i32>) -> tensor<1xi32>
    %4 = stablehlo.reshape %arg2 : (tensor<i32>) -> tensor<1xi32>
    %5 = stablehlo.concatenate %3, %4, dim = 0 : (tensor<1xi32>, tensor<1xi32>) -> tensor<2xi32>
    %6 = stablehlo.custom_call @tpu_custom_call(%0#0, %1#0, %2#0, %arg3, %arg4, %5) {backend_config = "{\22custom_call_config\22: {\22body\22: \22TUzvUgFNTElSMjAuMC4wZ2l0AAEtCQEDBQcBAwkDGQsNDxETFRcZGx0fIQP3zxMBxwcTDwsLEw8LCwsPDw8PCwtTCw8PCwsLDwsPDwsbDw8bCw9DCxcLhQtzCwsLCxcbCxsLGwsbGxsPCw8PEw8LEw8LExMPCw8TCxMPDwsPExMfCyMLCwsTCw8LC1MLExMPCw8LEwUHkWEqAgcDWQETDwcfJw8HCycjAmoJHwMDE5UdlxUFIwUlAwMTbx2PJwUnBSkFKxWZJR03FR03PREJAQUtBS8jCQUhgAAAAAAAAACAAAAAAAAAAA0RFXl/FYslBTEFMwU1HXFzBTcdoRURDQAFOQMFs7W3uR2/PRXBJwMFQUMJDwU7EQkNAw9HSRFLT1FTG1UbCVdZWwU9AQfHx8cND2FmZmluZV9tYXA8KGQwLCBkMSkgLT4gKGQwLCBkMSk+AAU/IwkHMQAAAAAAAACAAAAAAAAAAIAAAAAAAAAAgAVBBUMFRQVHAQddYWUDBR1fHyEJKQMFHWMfIQkrAwUdZx8hCS0DBREjCSkDBREjCSsDBREjCS0RAQEFSRV1JR0PdxcHJQEde30FSxcHaQEdgYMFTRcHhQEDAzEbHYknBU8dD40XBycBBVEDAzGTEQkFEQMBBVMdD5sXBy0BAwMTnyUFCQAAAAAFVQMHpcunNak1BVcFWQVbAwOtzQVdHbEVBV8FYSMBCSEBAAAAAQAAAAIAAAAAAAAABWMjAQEBAwMTvRMLAQVlHcPFBWcXBykBI3RwdS5kaW1lbnNpb25fc2VtYW50aWNzPGFyYml0cmFyeT4AI3RwdS5tZW1vcnlfc3BhY2U8dm1lbT4AI3RwdS5kb3RfZGltZW5zaW9uX251bWJlcnM8WzFdLCBbMF0sIFswXSwgWzFdLCBbMCwgMCwgMSwgMV0sIFtdLCBbXT4AI2FyaXRoLmZhc3RtYXRoPG5vbmU+AAECAgMnBQIEAgQLF8kFAgQCBAtNAQIECwEJBQ0BAQEHBwcBBQcBAQEFAQEEKgUFAREBPwcDAREHEQFFBwM1Xw0BAQEBAQEHAQcBBwEDAy8LAwELBy+FAw0FBQ0RBocDAQMPAwMNCwMBCwcNkQMNBRETExQNAxUJAwsdAwM7uwMLGQY7AwUDNQMDGQMDAwMDGQMDAwUGGQMFBws5Ow0FGTkJNws5Ow8ADQMBBQ8ADQMDBQMDAwMDBQMDAwUGBQMFBwsXGQMDBQMDAwMDBQMDAwUGBQMFBwcdHwMDBQMDAwMDBQMDAwUGBQMFBwkjJQMDM50DBRUHM6MDBQchJykXB6+rAwUFGysDAxcDAwMDAxcDAwMFBhcDBQcLLzENBRc5CS0LLzEJAAEHEQFpBwMJCwcBAQEBAQEDAwELAwEJBAEFAQUHEQFrBwMJCwcBAQEBAQEDAwELAwEJBAEFBQMHEQFtBwMJCwcBAQEBAQEDAwELAwEJBAEFAQMGAwEFAQCmCmkFJREpCxMdHSUbCw0tEw8JHQsjISMpLQ0VGRkZHSUNHR0TQSMXFw8ZFSMXGRUZHw8NCR0RYnVpbHRpbgBzdGFibGVfbW9zYWljAHRwdQBhcml0aABtb2R1bGUAYXJpdGguY29uc3RhbnQAdmVjdG9yLmxvYWQAZnVuYy5mdW5jAGZ1bmMucmV0dXJuAGFyaXRoLmNtcGkAdHB1LnZlY3Rvcl9zdG9yZQBzY2YueWllbGQAYXJpdGguZXh0dWkAc2NmLmlmAHRwdS5tYXRtdWwAYXJpdGguYWRkZgB2ZWN0b3IuYnJvYWRjYXN0AC9tbnQvY2hlbmdqaS9qYXgvZHluYW1pYy9lMmUucHkAc3ltX25hbWUAbWF0bXVsX2tlcm5lbABmdW5jdGlvbl90eXBlAHZhbHVlAHRyYW5zZm9ybV9pbmRpY2VzAHdpbmRvd19ib3VuZHMAdHJhbnNmb3JtXzAAdHJhbnNmb3JtXzEAdHJhbnNmb3JtXzIAcHJlZGljYXRlAC9zd2FwAHN0YWJsZV9tb3NhaWMudmVyc2lvbgBkaW1lbnNpb25fc2VtYW50aWNzAGl0ZXJhdGlvbl9ib3VuZHMAc2NhbGFyX3ByZWZldGNoAHNjcmF0Y2hfb3BlcmFuZHMAbWFpbgB3aW5kb3dfcGFyYW1zAC9lcQBtYXRtdWwAPG1vZHVsZT4AL2NvbnZlcnRfZWxlbWVudF90eXBlAC9jb25kAC9nZXQAL2RvdF9nZW5lcmFsAGRpbWVuc2lvbl9udW1iZXJzAHRyYW5zcG9zZV9saHMAdHJhbnNwb3NlX3JocwBmYXN0bWF0aAAvYWRkAG9wZXJhbmRTZWdtZW50U2l6ZXMAc3RyaWRlcwAvYnJvYWRjYXN0X2luX2RpbQBfAA==\22, \22serialization_format\22: 1, \22needs_layout_passes\22: true}}", indices_of_shape_operands = dense<5> : tensor<1xi64>, kernel_name = "matmul_kernel", operand_layouts = [dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<> : tensor<0xindex>, dense<[1, 0]> : tensor<2xindex>, dense<[1, 0]> : tensor<2xindex>, dense<0> : tensor<1xindex>], result_layouts = [dense<[1, 0]> : tensor<2xindex>]} : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>, tensor<2xi32>) -> tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>
    return %6 : tensor<?x?xf32, #stablehlo.bounds<2048, 2048>>
  }
  func.func private @divmod(%arg0: tensor<i32> {jax.global_constant = "l"}, %arg1: tensor<i32> {jax.global_constant = "m"}, %arg2: tensor<i32> {jax.global_constant = "n"}, %arg3: tensor<i32>, %arg4: tensor<i32>) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.convert %arg3 : tensor<i32>
    %1 = call @floor_divide(%arg0, %arg1, %arg2, %0, %arg4) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = call @remainder(%arg0, %arg1, %arg2, %0, %arg4) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %1, %2 : tensor<i32>, tensor<i32>
  }
  func.func private @floor_divide(%arg0: tensor<i32> {jax.global_constant = "l"}, %arg1: tensor<i32> {jax.global_constant = "m"}, %arg2: tensor<i32> {jax.global_constant = "n"}, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.divide %arg3, %arg4 : tensor<i32>
    %1 = stablehlo.sign %arg3 : tensor<i32>
    %2 = stablehlo.sign %arg4 : tensor<i32>
    %3 = stablehlo.compare  NE, %1, %2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = stablehlo.remainder %arg3, %arg4 : tensor<i32>
    %5 = stablehlo.compare  NE, %4, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.and %3, %5 : tensor<i1>
    %7 = stablehlo.subtract %0, %c : tensor<i32>
    %8 = call @_where(%arg0, %arg1, %arg2, %6, %7, %0) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %8 : tensor<i32>
  }
  func.func private @_where(%arg0: tensor<i32> {jax.global_constant = "l"}, %arg1: tensor<i32> {jax.global_constant = "m"}, %arg2: tensor<i32> {jax.global_constant = "n"}, %arg3: tensor<i1>, %arg4: tensor<i32>, %arg5: tensor<i32>) -> tensor<i32> {
    %0 = stablehlo.select %arg3, %arg4, %arg5 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @remainder(%arg0: tensor<i32> {jax.global_constant = "l"}, %arg1: tensor<i32> {jax.global_constant = "m"}, %arg2: tensor<i32> {jax.global_constant = "n"}, %arg3: tensor<i32>, %arg4: tensor<i32>) -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg4, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = call @_where(%arg0, %arg1, %arg2, %0, %c, %arg4) : (tensor<i32>, tensor<i32>, tensor<i32>, tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.remainder %arg3, %1 : tensor<i32>
    %3 = stablehlo.compare  NE, %2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = stablehlo.compare  LT, %2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.compare  LT, %1, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.compare  NE, %4, %5,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = stablehlo.and %6, %3 : tensor<i1>
    %8 = stablehlo.add %2, %1 : tensor<i32>
    %9 = stablehlo.select %7, %8, %2 : tensor<i1>, tensor<i32>
    return %9 : tensor<i32>
  }
  func.func private @divmod_0(%arg0: tensor<i32> {jax.global_constant = ""}, %arg1: tensor<i32> {jax.global_constant = ""}) -> (tensor<i32>, tensor<i32>) {
    %0 = stablehlo.convert %arg0 : tensor<i32>
    %1 = call @floor_divide_1(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = call @remainder_3(%0, %arg1) : (tensor<i32>, tensor<i32>) -> tensor<i32>
    return %1, %2 : tensor<i32>, tensor<i32>
  }
  func.func private @floor_divide_1(%arg0: tensor<i32> {jax.global_constant = ""}, %arg1: tensor<i32> {jax.global_constant = ""}) -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.divide %arg0, %arg1 : tensor<i32>
    %1 = stablehlo.sign %arg0 : tensor<i32>
    %2 = stablehlo.sign %arg1 : tensor<i32>
    %3 = stablehlo.compare  NE, %1, %2,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = stablehlo.remainder %arg0, %arg1 : tensor<i32>
    %5 = stablehlo.compare  NE, %4, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.and %3, %5 : tensor<i1>
    %7 = stablehlo.subtract %0, %c : tensor<i32>
    %8 = call @_where_2(%6, %7, %0) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    return %8 : tensor<i32>
  }
  func.func private @_where_2(%arg0: tensor<i1> {jax.global_constant = ""}, %arg1: tensor<i32> {jax.global_constant = ""}, %arg2: tensor<i32> {jax.global_constant = ""}) -> tensor<i32> {
    %0 = stablehlo.select %arg0, %arg1, %arg2 : tensor<i1>, tensor<i32>
    return %0 : tensor<i32>
  }
  func.func private @remainder_3(%arg0: tensor<i32> {jax.global_constant = ""}, %arg1: tensor<i32> {jax.global_constant = ""}) -> tensor<i32> {
    %c = stablehlo.constant dense<1> : tensor<i32>
    %c_0 = stablehlo.constant dense<0> : tensor<i32>
    %0 = stablehlo.compare  EQ, %arg1, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %1 = call @_where_2(%0, %c, %arg1) : (tensor<i1>, tensor<i32>, tensor<i32>) -> tensor<i32>
    %2 = stablehlo.remainder %arg0, %1 : tensor<i32>
    %3 = stablehlo.compare  NE, %2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %4 = stablehlo.compare  LT, %2, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %5 = stablehlo.compare  LT, %1, %c_0,  SIGNED : (tensor<i32>, tensor<i32>) -> tensor<i1>
    %6 = stablehlo.compare  NE, %4, %5,  UNSIGNED : (tensor<i1>, tensor<i1>) -> tensor<i1>
    %7 = stablehlo.and %6, %3 : tensor<i1>
    %8 = stablehlo.add %2, %1 : tensor<i32>
    %9 = stablehlo.select %7, %8, %2 : tensor<i1>, tensor<i32>
    return %9 : tensor<i32>
  }
}