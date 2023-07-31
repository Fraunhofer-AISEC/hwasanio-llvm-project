; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt -S -passes=sroa < %s | FileCheck %s

%pair = type { i32, i32 }

define i32 @test_sroa_phi_gep(i1 %cond) {
; CHECK-LABEL: @test_sroa_phi_gep(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI_SROA_PHI_SROA_SPECULATED:%.*]] = phi i32 [ 1, [[ENTRY:%.*]] ], [ 2, [[IF_THEN]] ]
; CHECK-NEXT:    ret i32 [[PHI_SROA_PHI_SROA_SPECULATED]]
;
entry:
  %a = alloca %pair, align 4
  %b = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  %gep_b = getelementptr inbounds %pair, %pair* %b, i32 0, i32 1
  store i32 1, i32* %gep_a, align 4
  store i32 2, i32* %gep_b, align 4
  br i1 %cond, label %if.then, label %end

if.then:
  br label %end

end:
  %phi = phi %pair* [ %a, %entry], [ %b, %if.then ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

define i32 @test_sroa_phi_gep_non_inbound(i1 %cond) {
; CHECK-LABEL: @test_sroa_phi_gep_non_inbound(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI_SROA_PHI_SROA_SPECULATED:%.*]] = phi i32 [ 1, [[ENTRY:%.*]] ], [ 2, [[IF_THEN]] ]
; CHECK-NEXT:    ret i32 [[PHI_SROA_PHI_SROA_SPECULATED]]
;
entry:
  %a = alloca %pair, align 4
  %b = alloca %pair, align 4
  %gep_a = getelementptr %pair, %pair* %a, i32 0, i32 1
  %gep_b = getelementptr %pair, %pair* %b, i32 0, i32 1
  store i32 1, i32* %gep_a, align 4
  store i32 2, i32* %gep_b, align 4
  br i1 %cond, label %if.then, label %end

if.then:
  br label %end

end:
  %phi = phi %pair* [ %a, %entry], [ %b, %if.then ]
  %gep = getelementptr %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

define i32 @test_sroa_phi_gep_undef(i1 %cond) {
; CHECK-LABEL: @test_sroa_phi_gep_undef(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [[PAIR:%.*]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[A]], [[ENTRY:%.*]] ], [ undef, [[IF_THEN]] ]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i32 0, i32 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
entry:
  %a = alloca %pair, align 4
  br i1 %cond, label %if.then, label %end

if.then:
  br label %end

end:
  %phi = phi %pair* [ %a, %entry], [ undef, %if.then ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

@g = global %pair zeroinitializer, align 4

define i32 @test_sroa_phi_gep_global(i1 %cond) {
; CHECK-LABEL: @test_sroa_phi_gep_global(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [[PAIR:%.*]], align 4
; CHECK-NEXT:    [[GEP_A:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[A]], i32 0, i32 1
; CHECK-NEXT:    store i32 1, i32* [[GEP_A]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[A]], [[ENTRY:%.*]] ], [ @g, [[IF_THEN]] ]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i32 0, i32 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
entry:
  %a = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  store i32 1, i32* %gep_a, align 4
  br i1 %cond, label %if.then, label %end

if.then:
  br label %end

end:
  %phi = phi %pair* [ %a, %entry], [ @g, %if.then ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

define i32 @test_sroa_phi_gep_arg_phi_inspt(i1 %cond) {
; CHECK-LABEL: @test_sroa_phi_gep_arg_phi_inspt(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[FOR:%.*]], label [[END:%.*]]
; CHECK:       for:
; CHECK-NEXT:    [[PHI_INSPT:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[I:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[I]] = add i32 [[PHI_INSPT]], 1
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[I]], 10
; CHECK-NEXT:    br i1 [[LOOP_COND]], label [[FOR]], label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI_SROA_PHI_SROA_SPECULATED:%.*]] = phi i32 [ 1, [[ENTRY]] ], [ 2, [[FOR]] ]
; CHECK-NEXT:    ret i32 [[PHI_SROA_PHI_SROA_SPECULATED]]
;
entry:
  %a = alloca %pair, align 4
  %b = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  %gep_b = getelementptr inbounds %pair, %pair* %b, i32 0, i32 1
  store i32 1, i32* %gep_a, align 4
  store i32 2, i32* %gep_b, align 4
  br i1 %cond, label %for, label %end

for:
  %phi_inspt = phi i32 [ 0, %entry ], [ %i, %for ]
  %i = add i32 %phi_inspt, 1
  %loop.cond = icmp ult i32 %i, 10
  br i1 %loop.cond, label %for, label %end

end:
  %phi = phi %pair* [ %a, %entry], [ %b, %for ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

define i32 @test_sroa_phi_gep_phi_inspt(i1 %cond) {
; CHECK-LABEL: @test_sroa_phi_gep_phi_inspt(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [[PAIR:%.*]], align 4
; CHECK-NEXT:    [[B:%.*]] = alloca [[PAIR]], align 4
; CHECK-NEXT:    [[GEP_A:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[A]], i32 0, i32 1
; CHECK-NEXT:    [[GEP_B:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[B]], i32 0, i32 1
; CHECK-NEXT:    store i32 1, i32* [[GEP_A]], align 4
; CHECK-NEXT:    store i32 2, i32* [[GEP_B]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[FOR:%.*]], label [[END:%.*]]
; CHECK:       for:
; CHECK-NEXT:    [[PHI_IN:%.*]] = phi %pair* [ null, [[ENTRY:%.*]] ], [ [[B]], [[FOR]] ]
; CHECK-NEXT:    [[PHI_INSPT:%.*]] = phi i32 [ 0, [[ENTRY]] ], [ [[I:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[I]] = add i32 [[PHI_INSPT]], 1
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[I]], 10
; CHECK-NEXT:    br i1 [[LOOP_COND]], label [[FOR]], label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[A]], [[ENTRY]] ], [ [[PHI_IN]], [[FOR]] ]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i32 0, i32 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
entry:
  %a = alloca %pair, align 4
  %b = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  %gep_b = getelementptr inbounds %pair, %pair* %b, i32 0, i32 1
  store i32 1, i32* %gep_a, align 4
  store i32 2, i32* %gep_b, align 4
  br i1 %cond, label %for, label %end

for:
  %phi_in = phi %pair * [ null, %entry ], [ %b, %for ]
  %phi_inspt = phi i32 [ 0, %entry ], [ %i, %for ]
  %i = add i32 %phi_inspt, 1
  %loop.cond = icmp ult i32 %i, 10
  br i1 %loop.cond, label %for, label %end

end:
  %phi = phi %pair* [ %a, %entry], [ %phi_in, %for ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

define i32 @test_sroa_gep_phi_gep(i1 %cond) {
; CHECK-LABEL: @test_sroa_gep_phi_gep(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A_SROA_0:%.*]] = alloca i32, align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[FOR:%.*]], label [[END:%.*]]
; CHECK:       for:
; CHECK-NEXT:    [[PHI_I:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[I:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[PHI:%.*]] = phi i32* [ [[A_SROA_0]], [[ENTRY]] ], [ [[GEP_FOR:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[I]] = add i32 [[PHI_I]], 1
; CHECK-NEXT:    [[GEP_FOR]] = getelementptr inbounds i32, i32* [[PHI]], i32 0
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[I]], 10
; CHECK-NEXT:    br i1 [[LOOP_COND]], label [[FOR]], label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI_END:%.*]] = phi i32* [ [[A_SROA_0]], [[ENTRY]] ], [ [[PHI]], [[FOR]] ]
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[PHI_END]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
entry:
  %a = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  br i1 %cond, label %for, label %end

for:
  %phi_i = phi i32 [ 0, %entry ], [ %i, %for ]
  %phi = phi i32* [ %gep_a, %entry], [ %gep_for, %for ]
  %i = add i32 %phi_i, 1
  %gep_for = getelementptr inbounds i32, i32* %phi, i32 0
  %loop.cond = icmp ult i32 %i, 10
  br i1 %loop.cond, label %for, label %end

end:
  %phi_end = phi i32* [ %gep_a, %entry], [ %phi, %for ]
  %load = load i32, i32* %phi_end, align 4
  ret i32 %load
}

define i32 @test_sroa_invoke_phi_gep(i1 %cond) personality i32 (...)* @__gxx_personality_v0 {
; CHECK-LABEL: @test_sroa_invoke_phi_gep(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [[PAIR:%.*]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[CALL:%.*]], label [[END:%.*]]
; CHECK:       call:
; CHECK-NEXT:    [[B:%.*]] = invoke %pair* @foo()
; CHECK-NEXT:    to label [[END]] unwind label [[INVOKE_CATCH:%.*]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[A]], [[ENTRY:%.*]] ], [ [[B]], [[CALL]] ]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i32 0, i32 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
; CHECK:       invoke_catch:
; CHECK-NEXT:    [[RES:%.*]] = landingpad { i8*, i32 }
; CHECK-NEXT:    catch i8* null
; CHECK-NEXT:    ret i32 0
;
entry:
  %a = alloca %pair, align 4
  br i1 %cond, label %call, label %end

call:
  %b = invoke %pair* @foo()
  to label %end unwind label %invoke_catch

end:
  %phi = phi %pair* [ %a, %entry], [ %b, %call ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load

invoke_catch:
  %res = landingpad { i8*, i32 }
  catch i8* null
  ret i32 0
}

define i32 @test_sroa_phi_gep_nonconst_idx(i1 %cond, i32 %idx) {
; CHECK-LABEL: @test_sroa_phi_gep_nonconst_idx(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A:%.*]] = alloca [[PAIR:%.*]], align 4
; CHECK-NEXT:    [[B:%.*]] = alloca [[PAIR]], align 4
; CHECK-NEXT:    [[GEP_A:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[A]], i32 0, i32 1
; CHECK-NEXT:    [[GEP_B:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[B]], i32 0, i32 1
; CHECK-NEXT:    store i32 1, i32* [[GEP_A]], align 4
; CHECK-NEXT:    store i32 2, i32* [[GEP_B]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[IF_THEN:%.*]], label [[END:%.*]]
; CHECK:       if.then:
; CHECK-NEXT:    br label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[A]], [[ENTRY:%.*]] ], [ [[B]], [[IF_THEN]] ]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i32 [[IDX:%.*]], i32 1
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[GEP]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
entry:
  %a = alloca %pair, align 4
  %b = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  %gep_b = getelementptr inbounds %pair, %pair* %b, i32 0, i32 1
  store i32 1, i32* %gep_a, align 4
  store i32 2, i32* %gep_b, align 4
  br i1 %cond, label %if.then, label %end

if.then:
  br label %end

end:
  %phi = phi %pair* [ %a, %entry], [ %b, %if.then ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 %idx, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

define void @test_sroa_gep_phi_select_other_block() {
; CHECK-LABEL: @test_sroa_gep_phi_select_other_block(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [[PAIR:%.*]], align 8
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.body:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[ALLOCA]], [[ENTRY:%.*]] ], [ [[SELECT:%.*]], [[WHILE_BODY]] ]
; CHECK-NEXT:    [[SELECT]] = select i1 undef, %pair* [[PHI]], %pair* undef
; CHECK-NEXT:    br i1 undef, label [[EXIT:%.*]], label [[WHILE_BODY]]
; CHECK:       exit:
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i64 1
; CHECK-NEXT:    unreachable
;
entry:
  %alloca = alloca %pair, align 8
  br label %while.body

while.body:
  %phi = phi %pair* [ %alloca, %entry ], [ %select, %while.body ]
  %select = select i1 undef, %pair* %phi, %pair* undef
  br i1 undef, label %exit, label %while.body

exit:
  %gep = getelementptr inbounds %pair, %pair* %phi, i64 1
  unreachable
}

define void @test_sroa_gep_phi_select_same_block() {
; CHECK-LABEL: @test_sroa_gep_phi_select_same_block(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[ALLOCA:%.*]] = alloca [[PAIR:%.*]], align 8
; CHECK-NEXT:    br label [[WHILE_BODY:%.*]]
; CHECK:       while.body:
; CHECK-NEXT:    [[PHI:%.*]] = phi %pair* [ [[ALLOCA]], [[ENTRY:%.*]] ], [ [[SELECT:%.*]], [[WHILE_BODY]] ]
; CHECK-NEXT:    [[SELECT]] = select i1 undef, %pair* [[PHI]], %pair* undef
; CHECK-NEXT:    [[PHI_SROA_GEP:%.*]] = getelementptr inbounds [[PAIR]], %pair* [[PHI]], i64 1
; CHECK-NEXT:    [[SELECT_SROA_SEL:%.*]] = select i1 undef, %pair* [[PHI_SROA_GEP]], %pair* poison
; CHECK-NEXT:    br i1 undef, label [[EXIT:%.*]], label [[WHILE_BODY]]
; CHECK:       exit:
; CHECK-NEXT:    unreachable
;
entry:
  %alloca = alloca %pair, align 8
  br label %while.body

while.body:
  %phi = phi %pair* [ %alloca, %entry ], [ %select, %while.body ]
  %select = select i1 undef, %pair* %phi, %pair* undef
  %gep = getelementptr inbounds %pair, %pair* %select, i64 1
  br i1 undef, label %exit, label %while.body

exit:
  unreachable
}

define i32 @test_sroa_gep_cast_phi_gep(i1 %cond) {
; CHECK-LABEL: @test_sroa_gep_cast_phi_gep(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[A_SROA_0:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A_SROA_0_0_GEP_A_CAST_TO_FLOAT_SROA_CAST:%.*]] = bitcast i32* [[A_SROA_0]] to float*
; CHECK-NEXT:    [[A_SROA_0_0_GEP_A_CAST_TO_FLOAT_SROA_CAST2:%.*]] = bitcast i32* [[A_SROA_0]] to float*
; CHECK-NEXT:    [[A_SROA_0_0_GEP_SROA_CAST:%.*]] = bitcast i32* [[A_SROA_0]] to float*
; CHECK-NEXT:    store i32 1065353216, i32* [[A_SROA_0]], align 4
; CHECK-NEXT:    br i1 [[COND:%.*]], label [[FOR:%.*]], label [[END:%.*]]
; CHECK:       for:
; CHECK-NEXT:    [[PHI_I:%.*]] = phi i32 [ 0, [[ENTRY:%.*]] ], [ [[I:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[PHI:%.*]] = phi float* [ [[A_SROA_0_0_GEP_A_CAST_TO_FLOAT_SROA_CAST]], [[ENTRY]] ], [ [[GEP_FOR_CAST_TO_FLOAT:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[PHI_SROA_PHI:%.*]] = phi float* [ [[A_SROA_0_0_GEP_SROA_CAST]], [[ENTRY]] ], [ [[GEP_FOR_CAST_TO_FLOAT_SROA_GEP:%.*]], [[FOR]] ]
; CHECK-NEXT:    [[I]] = add i32 [[PHI_I]], 1
; CHECK-NEXT:    [[GEP_FOR_CAST:%.*]] = bitcast float* [[PHI_SROA_PHI]] to i32*
; CHECK-NEXT:    [[GEP_FOR_CAST_TO_FLOAT]] = bitcast i32* [[GEP_FOR_CAST]] to float*
; CHECK-NEXT:    [[GEP_FOR_CAST_TO_FLOAT_SROA_GEP]] = getelementptr inbounds float, float* [[GEP_FOR_CAST_TO_FLOAT]], i32 0
; CHECK-NEXT:    [[LOOP_COND:%.*]] = icmp ult i32 [[I]], 10
; CHECK-NEXT:    br i1 [[LOOP_COND]], label [[FOR]], label [[END]]
; CHECK:       end:
; CHECK-NEXT:    [[PHI_END:%.*]] = phi float* [ [[A_SROA_0_0_GEP_A_CAST_TO_FLOAT_SROA_CAST2]], [[ENTRY]] ], [ [[PHI]], [[FOR]] ]
; CHECK-NEXT:    [[PHI_END_1:%.*]] = bitcast float* [[PHI_END]] to i32*
; CHECK-NEXT:    [[LOAD:%.*]] = load i32, i32* [[PHI_END_1]], align 4
; CHECK-NEXT:    ret i32 [[LOAD]]
;
entry:
  %a = alloca %pair, align 4
  %gep_a = getelementptr inbounds %pair, %pair* %a, i32 0, i32 1
  %gep_a_cast_to_float = bitcast i32* %gep_a to float*
  store float 1.0, float* %gep_a_cast_to_float, align 4
  br i1 %cond, label %for, label %end

for:
  %phi_i = phi i32 [ 0, %entry ], [ %i, %for ]
  %phi = phi float* [ %gep_a_cast_to_float, %entry], [ %gep_for_cast_to_float, %for ]
  %i = add i32 %phi_i, 1
  %gep_for = getelementptr inbounds float, float* %phi, i32 0
  %gep_for_cast = bitcast float* %gep_for to i32*
  %gep_for_cast_to_float = bitcast i32* %gep_for_cast to float*
  %loop.cond = icmp ult i32 %i, 10
  br i1 %loop.cond, label %for, label %end

end:
  %phi_end = phi float* [ %gep_a_cast_to_float, %entry], [ %phi, %for ]
  %phi_end.1 = bitcast float* %phi_end to i32*
  %load = load i32, i32* %phi_end.1, align 4
  ret i32 %load
}

define void @unreachable_term() {
; CHECK-LABEL: @unreachable_term(
; CHECK-NEXT:    [[A_SROA_0:%.*]] = alloca i32, align 4
; CHECK-NEXT:    [[A_SROA_0_0_A_SROA_CAST1:%.*]] = bitcast i32* [[A_SROA_0]] to [3 x i32]*
; CHECK-NEXT:    unreachable
; CHECK:       bb1:
; CHECK-NEXT:    br label [[BB1_I:%.*]]
; CHECK:       bb1.i:
; CHECK-NEXT:    [[PHI:%.*]] = phi [3 x i32]* [ [[A_SROA_0_0_A_SROA_CAST1]], [[BB1:%.*]] ], [ null, [[BB1_I]] ]
; CHECK-NEXT:    [[GEP:%.*]] = getelementptr [3 x i32], [3 x i32]* [[PHI]], i64 0, i64 0
; CHECK-NEXT:    store i32 0, i32* [[GEP]], align 1
; CHECK-NEXT:    br i1 undef, label [[BB1_I]], label [[EXIT:%.*]]
; CHECK:       exit:
; CHECK-NEXT:    br label [[BB2:%.*]]
; CHECK:       bb2:
; CHECK-NEXT:    ret void
;
  %a = alloca [3 x i32], align 1
  unreachable

bb1:
  br label %bb1.i

bb1.i:
  %phi = phi [3 x i32]* [ %a, %bb1 ], [ null, %bb1.i ]
  %gep = getelementptr [3 x i32], [3 x i32]* %phi, i64 0, i64 0
  store i32 0, i32* %gep, align 1
  br i1 undef, label %bb1.i, label %exit

exit:
  br label %bb2

bb2:
  ret void
}

define void @constant_value_phi() {
; CHECK-LABEL: @constant_value_phi(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    br label [[LAND_LHS_TRUE_I:%.*]]
; CHECK:       land.lhs.true.i:
; CHECK-NEXT:    br i1 undef, label [[COND_END_I:%.*]], label [[COND_END_I]]
; CHECK:       cond.end.i:
; CHECK-NEXT:    unreachable
;
entry:
  %s1 = alloca [3 x i16]
  %s = alloca [3 x i16]
  %cast = bitcast [3 x i16]* %s1 to i16*
  br label %land.lhs.true.i

land.lhs.true.i:                                  ; preds = %entry
  br i1 undef, label %cond.end.i, label %cond.end.i

cond.end.i:                                       ; preds = %land.lhs.true.i, %land.lhs.true.i
  %.pre-phi1 = phi i16* [ %cast, %land.lhs.true.i ], [ %cast, %land.lhs.true.i ]
  %cast2 = bitcast [3 x i16]* %s to i16*
  call void @llvm.memcpy.p0i16.p0i16.i64(i16* %.pre-phi1, i16* %cast2, i64 3, i1 false)
  %gep = getelementptr inbounds [3 x i16], [3 x i16]* %s, i32 0, i32 0
  %load = load i16, i16* %gep
  unreachable
}

define i32 @test_sroa_phi_gep_multiple_values_from_same_block(i32 %arg) {
; CHECK-LABEL: @test_sroa_phi_gep_multiple_values_from_same_block(
; CHECK-NEXT:  bb.1:
; CHECK-NEXT:    switch i32 [[ARG:%.*]], label [[BB_3:%.*]] [
; CHECK-NEXT:    i32 1, label [[BB_2:%.*]]
; CHECK-NEXT:    i32 2, label [[BB_2]]
; CHECK-NEXT:    i32 3, label [[BB_4:%.*]]
; CHECK-NEXT:    i32 4, label [[BB_4]]
; CHECK-NEXT:    ]
; CHECK:       bb.2:
; CHECK-NEXT:    br label [[BB_4]]
; CHECK:       bb.3:
; CHECK-NEXT:    br label [[BB_4]]
; CHECK:       bb.4:
; CHECK-NEXT:    [[PHI_SROA_PHI_SROA_SPECULATED:%.*]] = phi i32 [ undef, [[BB_3]] ], [ undef, [[BB_2]] ], [ undef, [[BB_1:%.*]] ], [ undef, [[BB_1]] ]
; CHECK-NEXT:    ret i32 [[PHI_SROA_PHI_SROA_SPECULATED]]
;
bb.1:
  %a = alloca %pair, align 4
  %b = alloca %pair, align 4
  switch i32 %arg, label %bb.3 [
  i32 1, label %bb.2
  i32 2, label %bb.2
  i32 3, label %bb.4
  i32 4, label %bb.4
  ]

bb.2:                                                ; preds = %bb.1, %bb.1
  br label %bb.4

bb.3:                                                ; preds = %bb.1
  br label %bb.4

bb.4:                                                ; preds = %bb.1, %bb.1, %bb.3, %bb.2
  %phi = phi %pair* [ %a, %bb.3 ], [ %a, %bb.2 ], [ %b, %bb.1 ], [ %b, %bb.1 ]
  %gep = getelementptr inbounds %pair, %pair* %phi, i32 0, i32 1
  %load = load i32, i32* %gep, align 4
  ret i32 %load
}

declare %pair* @foo()

declare i32 @__gxx_personality_v0(...)

declare void @llvm.memcpy.p0i16.p0i16.i64(i16* noalias nocapture writeonly, i16* noalias nocapture readonly, i64, i1 immarg)