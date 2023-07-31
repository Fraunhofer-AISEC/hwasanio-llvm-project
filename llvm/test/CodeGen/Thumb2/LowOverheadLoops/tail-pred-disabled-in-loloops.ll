; NOTE: Assertions have been autogenerated by utils/update_llc_test_checks.py
;
; RUN: llc -mtriple=thumbv8.1m.main -mattr=+mve -tail-predication=enabled \
; RUN:     %s -o - --verify-machineinstrs | FileCheck %s --check-prefix=ENABLED
;
; RUN: llc -mtriple=thumbv8.1m.main -mattr=+mve -tail-predication=enabled \
; RUN:     -arm-loloops-disable-tailpred %s -o - --verify-machineinstrs | \
; RUN:     FileCheck %s --check-prefix=DISABLED

define dso_local void @check_option(i32* noalias nocapture %A, i32* noalias nocapture readonly %B, i32* noalias nocapture readonly %C, i32 %N) local_unnamed_addr #0 {
; ENABLED-LABEL: check_option:
; ENABLED:       @ %bb.0: @ %entry
; ENABLED-NEXT:    push.w {r4, r5, r6, r7, r8, lr}
; ENABLED-NEXT:    cmp r3, #1
; ENABLED-NEXT:    blt .LBB0_4
; ENABLED-NEXT:  @ %bb.1: @ %vector.ph.preheader
; ENABLED-NEXT:  .LBB0_2: @ %vector.ph
; ENABLED-NEXT:    @ =>This Loop Header: Depth=1
; ENABLED-NEXT:    @ Child Loop BB0_3 Depth 2
; ENABLED-NEXT:    mov r12, r0
; ENABLED-NEXT:    mov r4, r2
; ENABLED-NEXT:    mov r5, r1
; ENABLED-NEXT:    mov r6, r3
; ENABLED-NEXT:    dlstp.32 lr, r6
; ENABLED-NEXT:  .LBB0_3: @ %vector.body
; ENABLED-NEXT:    @ Parent Loop BB0_2 Depth=1
; ENABLED-NEXT:    @ => This Inner Loop Header: Depth=2
; ENABLED-NEXT:    vldrw.u32 q0, [r5], #16
; ENABLED-NEXT:    vldrw.u32 q1, [r4], #16
; ENABLED-NEXT:    vadd.i32 q0, q1, q0
; ENABLED-NEXT:    vstrw.32 q0, [r12], #16
; ENABLED-NEXT:    letp lr, .LBB0_3
; ENABLED-NEXT:    b .LBB0_2
; ENABLED-NEXT:  .LBB0_4: @ %for.cond.cleanup
; ENABLED-NEXT:    pop.w {r4, r5, r6, r7, r8, pc}
;
; DISABLED-LABEL: check_option:
; DISABLED:       @ %bb.0: @ %entry
; DISABLED-NEXT:    push.w {r4, r5, r6, r7, r8, lr}
; DISABLED-NEXT:    cmp r3, #1
; DISABLED-NEXT:    blt .LBB0_4
; DISABLED-NEXT:  @ %bb.1: @ %vector.ph.preheader
; DISABLED-NEXT:    adds r7, r3, #3
; DISABLED-NEXT:    movs r6, #1
; DISABLED-NEXT:    bic r7, r7, #3
; DISABLED-NEXT:    subs r7, #4
; DISABLED-NEXT:    add.w r8, r6, r7, lsr #2
; DISABLED-NEXT:  .LBB0_2: @ %vector.ph
; DISABLED-NEXT:    @ =>This Loop Header: Depth=1
; DISABLED-NEXT:    @ Child Loop BB0_3 Depth 2
; DISABLED-NEXT:    mov r7, r8
; DISABLED-NEXT:    mov r12, r0
; DISABLED-NEXT:    mov r4, r2
; DISABLED-NEXT:    mov r5, r1
; DISABLED-NEXT:    mov r6, r3
; DISABLED-NEXT:    dls lr, r8
; DISABLED-NEXT:  .LBB0_3: @ %vector.body
; DISABLED-NEXT:    @ Parent Loop BB0_2 Depth=1
; DISABLED-NEXT:    @ => This Inner Loop Header: Depth=2
; DISABLED-NEXT:    vctp.32 r6
; DISABLED-NEXT:    mov lr, r7
; DISABLED-NEXT:    vpstt
; DISABLED-NEXT:    vldrwt.u32 q0, [r5], #16
; DISABLED-NEXT:    vldrwt.u32 q1, [r4], #16
; DISABLED-NEXT:    subs r7, #1
; DISABLED-NEXT:    subs r6, #4
; DISABLED-NEXT:    vadd.i32 q0, q1, q0
; DISABLED-NEXT:    vpst
; DISABLED-NEXT:    vstrwt.32 q0, [r12], #16
; DISABLED-NEXT:    le lr, .LBB0_3
; DISABLED-NEXT:    b .LBB0_2
; DISABLED-NEXT:  .LBB0_4: @ %for.cond.cleanup
; DISABLED-NEXT:    pop.w {r4, r5, r6, r7, r8, pc}
entry:
  %cmp8 = icmp sgt i32 %N, 0
  %0 = add i32 %N, 3
  %1 = lshr i32 %0, 2
  %2 = shl nuw i32 %1, 2
  %3 = add i32 %2, -4
  %4 = lshr i32 %3, 2
  %5 = add nuw nsw i32 %4, 1
  br i1 %cmp8, label %vector.ph, label %for.cond.cleanup

vector.ph:                                        ; preds = %entry
  %start = call i32 @llvm.start.loop.iterations.i32(i32 %5)
  br label %vector.body

vector.body:                                      ; preds = %vector.body, %vector.ph
  %lsr.iv17 = phi i32* [ %scevgep18, %vector.body ], [ %A, %vector.ph ]
  %lsr.iv14 = phi i32* [ %scevgep15, %vector.body ], [ %C, %vector.ph ]
  %lsr.iv = phi i32* [ %scevgep, %vector.body ], [ %B, %vector.ph ]
  %index = phi i32 [ 0, %vector.ph ], [ %index.next, %vector.body ]
  %6 = phi i32 [ %start, %vector.ph ], [ %8, %vector.body ]
  %lsr.iv13 = bitcast i32* %lsr.iv to <4 x i32>*
  %lsr.iv1416 = bitcast i32* %lsr.iv14 to <4 x i32>*
  %lsr.iv1719 = bitcast i32* %lsr.iv17 to <4 x i32>*
  %active.lane.mask = call <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32 %index, i32 %N)
  %wide.masked.load = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv13, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %wide.masked.load12 = call <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>* %lsr.iv1416, i32 4, <4 x i1> %active.lane.mask, <4 x i32> undef)
  %7 = add nsw <4 x i32> %wide.masked.load12, %wide.masked.load
  call void @llvm.masked.store.v4i32.p0v4i32(<4 x i32> %7, <4 x i32>* %lsr.iv1719, i32 4, <4 x i1> %active.lane.mask)
  %index.next = add i32 %index, 4
  %scevgep = getelementptr i32, i32* %lsr.iv, i32 4
  %scevgep15 = getelementptr i32, i32* %lsr.iv14, i32 4
  %scevgep18 = getelementptr i32, i32* %lsr.iv17, i32 4
  %8 = call i32 @llvm.loop.decrement.reg.i32(i32 %6, i32 1)
  %9 = icmp ne i32 %8, 0
  ;br i1 %9, label %vector.body, label %for.cond.cleanup
  br i1 %9, label %vector.body, label %vector.ph

for.cond.cleanup:                                 ; preds = %vector.body, %entry
  ret void
}

declare <4 x i32> @llvm.masked.load.v4i32.p0v4i32(<4 x i32>*, i32 immarg, <4 x i1>, <4 x i32>)
declare void @llvm.masked.store.v4i32.p0v4i32(<4 x i32>, <4 x i32>*, i32 immarg, <4 x i1>)
declare i32 @llvm.start.loop.iterations.i32(i32)
declare i32 @llvm.loop.decrement.reg.i32(i32, i32)
declare <4 x i1> @llvm.get.active.lane.mask.v4i1.i32(i32, i32)