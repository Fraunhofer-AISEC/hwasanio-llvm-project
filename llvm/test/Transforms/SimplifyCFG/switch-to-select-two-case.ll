; NOTE: Assertions have been autogenerated by utils/update_test_checks.py
; RUN: opt < %s -simplifycfg -simplifycfg-require-and-preserve-domtree=1 -S | FileCheck %s

; int foo1_with_default(int a) {
;   switch(a) {
;     case 10:
;       return 10;
;     case 20:
;       return 2;
;   }
;   return 4;
; }

define i32 @foo1_with_default(i32 %a) {
; CHECK-LABEL: @foo1_with_default(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SWITCH_SELECTCMP:%.*]] = icmp eq i32 [[A:%.*]], 20
; CHECK-NEXT:    [[SWITCH_SELECT:%.*]] = select i1 [[SWITCH_SELECTCMP]], i32 2, i32 4
; CHECK-NEXT:    [[SWITCH_SELECTCMP1:%.*]] = icmp eq i32 [[A]], 10
; CHECK-NEXT:    [[SWITCH_SELECT2:%.*]] = select i1 [[SWITCH_SELECTCMP1]], i32 10, i32 [[SWITCH_SELECT]]
; CHECK-NEXT:    ret i32 [[SWITCH_SELECT2]]
;
entry:
  switch i32 %a, label %sw.epilog [
  i32 10, label %sw.bb
  i32 20, label %sw.bb1
  ]

sw.bb:
  br label %return

sw.bb1:
  br label %return

sw.epilog:
  br label %return

return:
  %retval.0 = phi i32 [ 4, %sw.epilog ], [ 2, %sw.bb1 ], [ 10, %sw.bb ]
  ret i32 %retval.0
}

; Same as above, but both cases have the same value.
define i32 @same_value(i32 %a) {
; CHECK-LABEL: @same_value(
; CHECK-NEXT:  entry:
; CHECK-NEXT:    [[SWITCH_SELECTCMP_CASE1:%.*]] = icmp eq i32 [[A:%.*]], 10
; CHECK-NEXT:    [[SWITCH_SELECTCMP_CASE2:%.*]] = icmp eq i32 [[A]], 20
; CHECK-NEXT:    [[SWITCH_SELECTCMP:%.*]] = or i1 [[SWITCH_SELECTCMP_CASE1]], [[SWITCH_SELECTCMP_CASE2]]
; CHECK-NEXT:    [[TMP0:%.*]] = select i1 [[SWITCH_SELECTCMP]], i32 10, i32 4
; CHECK-NEXT:    ret i32 [[TMP0]]
;
entry:
  switch i32 %a, label %sw.epilog [
  i32 10, label %sw.bb
  i32 20, label %sw.bb
  ]

sw.bb:
  br label %return

sw.epilog:
  br label %return

return:
  %retval.0 = phi i32 [ 4, %sw.epilog ], [ 10, %sw.bb ]
  ret i32 %retval.0
}