//===- HWAddressSanitizer.cpp - detector of uninitialized reads -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This file is a part of HWAddressSanitizer, an address basic correctness
/// checker based on tagged addressing.
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Instrumentation/HWAddressSanitizer.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Triple.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/StackSafetyAnalysis.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/BinaryFormat/ELF.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/TypeFinder.h"
#include "llvm/IR/Value.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/PassRegistry.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Instrumentation.h"
#include "llvm/Transforms/Instrumentation/AddressSanitizerCommon.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"
#include "llvm/Transforms/Utils/PromoteMemToReg.h"
#include <sstream>
#include <unordered_map>

using namespace llvm;

// #define DISABLE_STACK_INSTRUMENTATION
// #define DISABLE_SHADE_GLOBALS
// #define DISABLE_SHADING_ALLOCATIONS
// #define DISABLE_GEP_INSTRUMENTATION
// #define DISABLE_CHECKS
// #define DISABLE_SHADE_CHECKS
#define DEBUG_TYPE "hwasan"
#define DEBUG_CHECKS  // this passes a unique id for every check to the backend,
                      // so it's easier to debug
#define DEBUG_MODULES // this passes a unique id for every check to the backend,
                      // so it's easier to debug
// #define SHADE_BUFFERS_ONLY
#define FIND_ALLOC_WRAPPERS false

const char kHwasanModuleCtorName[] = "hwasan.module_ctor";
const char kHwasanNoteName[] = "hwasan.note";
const char kHwasanInitName[] = "__hwasan_init";
const char kHwasanPersonalityThunkName[] = "__hwasan_personality_thunk";

const char kHwasanShadowMemoryDynamicAddress[] =
    "__hwasan_shadow_memory_dynamic_address";

// Accesses sizes are powers of two: 1, 2, 4, 8, 16.
static const size_t kNumberOfAccessSizes = 5;
#define ONE_TO_ONE_MAPPING

#ifdef ONE_TO_ONE_MAPPING
static const size_t kDefaultShadowScale = 0;
#else
static const size_t kDefaultShadowScale = 4;
#endif
static const uint64_t kDynamicShadowSentinel =
    std::numeric_limits<uint64_t>::max();

static const unsigned kShadowBaseAlignment = 32;

static cl::opt<std::string>
    ClMemoryAccessCallbackPrefix("hwasan-memory-access-callback-prefix",
                                 cl::desc("Prefix for memory access callbacks"),
                                 cl::Hidden, cl::init("__hwasan_"));

static cl::opt<bool> ClInstrumentWithCalls(
    "hwasan-instrument-with-calls",
    cl::desc("instrument reads and writes with callbacks"), cl::Hidden,
    cl::init(true));

static cl::opt<bool> ClInstrumentReads("hwasan-instrument-reads",
                                       cl::desc("instrument read instructions"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClInstrumentWrites("hwasan-instrument-writes",
                       cl::desc("instrument write instructions"), cl::Hidden,
                       cl::init(true));

static cl::opt<bool> ClInstrumentAtomics(
    "hwasan-instrument-atomics",
    cl::desc("instrument atomic instructions (rmw, cmpxchg)"), cl::Hidden,
    cl::init(true));

static cl::opt<bool> ClInstrumentByval("hwasan-instrument-byval",
                                       cl::desc("instrument byval arguments"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClRecover("hwasan-recover",
              cl::desc("Enable recovery mode (continue-after-error)."),
              cl::Hidden, cl::init(false));

static cl::opt<bool> ClInstrumentStack("hwasan-instrument-stack",
                                       cl::desc("instrument stack (allocas)"),
                                       cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClUseStackSafety("hwasan-use-stack-safety", cl::Hidden, cl::init(true),
                     cl::Hidden, cl::desc("Use Stack Safety analysis results"),
                     cl::Optional);

static cl::opt<size_t> ClMaxLifetimes(
    "hwasan-max-lifetimes-for-alloca", cl::Hidden, cl::init(3),
    cl::ReallyHidden,
    cl::desc("How many lifetime ends to handle for a single alloca."),
    cl::Optional);

static cl::opt<bool>
    ClUseAfterScope("hwasan-use-after-scope",
                    cl::desc("detect use after scope within function"),
                    cl::Hidden, cl::init(false));

static cl::opt<bool> ClUARRetagToZero(
    "hwasan-uar-retag-to-zero",
    cl::desc("Clear alloca tags before returning from the function to allow "
             "non-instrumented and instrumented function calls mix. When set "
             "to false, allocas are retagged before returning from the "
             "function to detect use after return."),
    cl::Hidden, cl::init(true));

static cl::opt<bool> ClGenerateTagsWithCalls(
    "hwasan-generate-tags-with-calls",
    cl::desc("generate new tags with runtime library calls"), cl::Hidden,
    cl::init(false));

static cl::opt<bool> ClGlobals("hwasan-globals", cl::desc("Instrument globals"),
                               cl::Hidden, cl::init(false), cl::ZeroOrMore);

static cl::opt<int> ClMatchAllTag(
    "hwasan-match-all-tag",
    cl::desc("don't report bad accesses via pointers with this tag"),
    cl::Hidden, cl::init(-1));

static cl::opt<bool>
    ClEnableKhwasan("hwasan-kernel",
                    cl::desc("Enable KernelHWAddressSanitizer instrumentation"),
                    cl::Hidden, cl::init(false));

// These flags allow to change the shadow mapping and control how shadow memory
// is accessed. The shadow mapping looks like:
//    Shadow = (Mem >> scale) + offset

static cl::opt<uint64_t>
    ClMappingOffset("hwasan-mapping-offset",
                    cl::desc("HWASan shadow mapping offset [EXPERIMENTAL]"),
                    cl::Hidden, cl::init(0));

static cl::opt<bool>
    ClWithIfunc("hwasan-with-ifunc",
                cl::desc("Access dynamic shadow through an ifunc global on "
                         "platforms that support this"),
                cl::Hidden, cl::init(false));

static cl::opt<bool> ClWithTls(
    "hwasan-with-tls",
    cl::desc("Access dynamic shadow through an thread-local pointer on "
             "platforms that support this"),
    cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClRecordStackHistory("hwasan-record-stack-history",
                         cl::desc("Record stack frames with tagged allocations "
                                  "in a thread-local ring buffer"),
                         cl::Hidden, cl::init(true));
static cl::opt<bool>
    ClInstrumentMemIntrinsics("hwasan-instrument-mem-intrinsics",
                              cl::desc("instrument memory intrinsics"),
                              cl::Hidden, cl::init(true));

static cl::opt<bool>
    ClInstrumentLandingPads("hwasan-instrument-landing-pads",
                            cl::desc("instrument landing pads"), cl::Hidden,
                            cl::init(false), cl::ZeroOrMore);

static cl::opt<bool> ClUseShortGranules(
    "hwasan-use-short-granules",
    cl::desc("use short granules in allocas and outlined checks"), cl::Hidden,
    cl::init(false), cl::ZeroOrMore);

static cl::opt<bool> ClInstrumentPersonalityFunctions(
    "hwasan-instrument-personality-functions",
    cl::desc("instrument personality functions"), cl::Hidden, cl::init(false),
    cl::ZeroOrMore);

static cl::opt<bool> ClInlineAllChecks("hwasan-inline-all-checks",
                                       cl::desc("inline all checks"),
                                       cl::Hidden, cl::init(false));

// Enabled from clang by "-fsanitize-hwaddress-experimental-aliasing".
static cl::opt<bool> ClUsePageAliases("hwasan-experimental-use-page-aliases",
                                      cl::desc("Use page aliasing in HWASan"),
                                      cl::Hidden, cl::init(false));

namespace {

bool shouldUsePageAliases(const Triple &TargetTriple) {
  return ClUsePageAliases && TargetTriple.getArch() == Triple::x86_64;
}

bool shouldInstrumentStack(const Triple &TargetTriple) {
  return !shouldUsePageAliases(TargetTriple) && ClInstrumentStack;
}

bool shouldInstrumentWithCalls(const Triple &TargetTriple) {
  return ClInstrumentWithCalls || TargetTriple.getArch() == Triple::x86_64;
}

bool mightUseStackSafetyAnalysis(bool DisableOptimization) {
  return ClUseStackSafety.getNumOccurrences() ? ClUseStackSafety
                                              : !DisableOptimization;
}

bool shouldUseStackSafetyAnalysis(const Triple &TargetTriple,
                                  bool DisableOptimization) {
  return shouldInstrumentStack(TargetTriple) &&
         mightUseStackSafetyAnalysis(DisableOptimization);
}

bool shouldDetectUseAfterScope(const Triple &TargetTriple) {
  return ClUseAfterScope && shouldInstrumentStack(TargetTriple);
}

/// An instrumentation pass implementing detection of addressability bugs
/// using tagged pointers.
class HWAddressSanitizer {
private:
  struct AllocaInfo {
    AllocaInst *AI;
    SmallVector<IntrinsicInst *, 2> LifetimeStart;
    SmallVector<IntrinsicInst *, 2> LifetimeEnd;
  };

public:
  HWAddressSanitizer(Module &M, bool CompileKernel, bool Recover,
                     const StackSafetyGlobalInfo *SSI)
      : M(M), SSI(SSI) {
    this->Recover = ClRecover.getNumOccurrences() > 0 ? ClRecover : Recover;
    this->CompileKernel = ClEnableKhwasan.getNumOccurrences() > 0
                              ? ClEnableKhwasan
                              : CompileKernel;
    DL = &(M.getDataLayout());
    initializeModule();
  }

  void setSSI(const StackSafetyGlobalInfo *S) { SSI = S; }

  DenseMap<AllocaInst *, AllocaInst *> padInterestingAllocas(
      const MapVector<AllocaInst *, AllocaInfo> &AllocasToInstrument);
  bool sanitizeFunction(Function &F,
                        llvm::function_ref<const DominatorTree &()> GetDT,
                        llvm::function_ref<const PostDominatorTree &()> GetPDT);
  bool CallNeedsInstrumentation(const CallBase *CI);
  void initializeFunctionDefinitionList(Module &M);

  void initializeModule();
  void createHwasanCtorComdat();

  void initializeCallbacks(Module &M);

  Value *getOpaqueNoopCast(IRBuilder<> &IRB, Value *Val);

  Value *getDynamicShadowIfunc(IRBuilder<> &IRB);
  Value *getShadowNonTls(IRBuilder<> &IRB);

  bool isAllocationFunction(CallBase *CB);
  bool isAllocationWrapper(CallBase *CB);

  bool isNotStruct(Type *T);
  bool isNotPartOfStruct(Value *I);
  void untagPointerOperand(Instruction *I, Value *Addr);
  Value *memToShadow(Value *Shadow, IRBuilder<> &IRB);
  Value *memToShade(Value *Shadow, IRBuilder<> &IRB);

  int64_t getAccessInfo(bool IsWrite, unsigned AccessSizeIndex);
  void instrumentMemAccessOutline(Value *Ptr, bool IsWrite,
                                  unsigned AccessSizeIndex,
                                  Instruction *InsertBefore);
  void instrumentMemAccessInline(Value *Ptr, bool IsWrite,
                                 unsigned AccessSizeIndex,
                                 Instruction *InsertBefore);
  bool ignoreMemIntrinsic(MemIntrinsic *MI);
  void instrumentMemIntrinsic(MemIntrinsic *MI);
  bool instrumentMemAccess(InterestingMemoryOperand &O);
  bool instrumentGEP(GetElementPtrInst *GEPI);
  bool instrumentCMP(Instruction *CMPI);
  bool instrumentConstGEP(ConstantExpr *ConstGEPI);
  bool containsUnion(StructType *type);
  std::vector<std::tuple<int, int>> getMetadataLayoutFromType(StructType *type);
  void instrumentMalloc(CallInst *GEPI);
  bool ignoreAccess(Instruction *Inst, Value *Ptr);
  void getInterestingMemoryOperands(
      Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting);

  bool isInterestingAlloca(const AllocaInst &AI);
  void tagAlloca(IRBuilder<> &IRB, AllocaInst *AI, Value *Tag, size_t Size);
  void shadeAlloca(IRBuilder<> &IRB, AllocaInst *AI, Value *Tag, size_t Size);
  void applyShading(IRBuilder<> &IRB, StructType *type, Value *AllocaInst);
  Value *tagPointer(IRBuilder<> &IRB, Type *Ty, Value *PtrLong, Value *Tag);
  Value *reshadePointer(IRBuilder<> &IRB, Type *Ty, Value *PtrLong,
                        Value *Shade);
  Constant *reshadeGlobalPointer(Type *Ty, Constant *PtrLong, Constant *Shade);
  Value *untagPointer(IRBuilder<> &IRB, Value *PtrLong);
  static bool isStandardLifetime(const AllocaInfo &AllocaInfo,
                                 const DominatorTree &DT);
  bool instrumentStack(
      bool ShouldDetectUseAfterScope,
      MapVector<AllocaInst *, AllocaInfo> &AllocasToInstrument,
      SmallVector<Instruction *, 4> &UnrecognizedLifetimes,
      DenseMap<AllocaInst *, std::vector<DbgVariableIntrinsic *>> &AllocaDbgMap,
      SmallVectorImpl<Instruction *> &RetVec, Value *StackTag,
      llvm::function_ref<const DominatorTree &()> GetDT,
      llvm::function_ref<const PostDominatorTree &()> GetPDT);
  Value *readRegister(IRBuilder<> &IRB, StringRef Name);
  bool instrumentLandingPads(SmallVectorImpl<Instruction *> &RetVec);
  Value *getNextTagWithCall(IRBuilder<> &IRB);
  Value *getStackBaseTag(IRBuilder<> &IRB);
  Value *getAllocaTag(IRBuilder<> &IRB, Value *StackTag, AllocaInst *AI,
                      unsigned AllocaNo);
  Value *getUARTag(IRBuilder<> &IRB, Value *StackTag);

  Value *getHwasanThreadSlotPtr(IRBuilder<> &IRB, Type *Ty);
  Value *applyTagMask(IRBuilder<> &IRB, Value *OldTag);
  unsigned retagMask(unsigned AllocaNo);

  void emitPrologue(IRBuilder<> &IRB, bool WithFrameRecord);

  void instrumentGlobal(GlobalVariable *GV, uint8_t Tag);
  void instrumentGlobals();

  void instrumentPersonalityFunctions();
  std::vector<GlobalVariable *> Globals;
  std::set<GlobalVariable::GUID> FunctionDefintionSet;
  std::vector<GlobalVariable *> StructGlobals;
  std::unordered_map<StructType *, std::vector<std::tuple<int, int>>>
      structTypeToBufferMap;
  SmallVector<ConstantExpr *, 4> ConstGEPsToInstrument;

private:
  LLVMContext *C;
  Module &M;
  const StackSafetyGlobalInfo *SSI;
  Triple TargetTriple;
  FunctionCallee HWAsanMemmove, HWAsanMemcpy, HWAsanMemset;
  FunctionCallee HWAsanHandleVfork;

  /// This struct defines the shadow mapping using the rule:
  ///   shadow = (mem >> Scale) + Offset.
  /// If InGlobal is true, then
  ///   extern char __hwasan_shadow[];
  ///   shadow = (mem >> Scale) + &__hwasan_shadow
  /// If InTls is true, then
  ///   extern char *__hwasan_tls;
  ///   shadow = (mem>>Scale) + align_up(__hwasan_shadow,
  ///   kShadowBaseAlignment)
  ///
  /// If WithFrameRecord is true, then __hwasan_tls will be used to access the
  /// ring buffer for storing stack allocations on targets that support it.
  struct ShadowMapping {
    int Scale;
    uint64_t Offset;
    bool InGlobal;
    bool InTls;
    bool WithFrameRecord;

    void init(Triple &TargetTriple, bool InstrumentWithCalls);
    uint64_t getObjectAlignment() const { return 1ULL << Scale; }
  };

  ShadowMapping Mapping;

  const DataLayout *DL;
  Type *VoidTy = Type::getVoidTy(M.getContext());
  Type *IntptrTy;
  Type *Int8PtrTy;
  Type *Int8Ty;
  Type *Int32Ty;
  Type *Int64Ty = Type::getInt64Ty(M.getContext());

  bool CompileKernel;
  bool Recover;
  bool OutlinedChecks;
  bool UseShortGranules;
  bool InstrumentLandingPads;
  bool InstrumentWithCalls;
  bool InstrumentStack;
  bool DetectUseAfterScope;
  bool UsePageAliases;

  bool HasMatchAllTag = false;
  uint8_t MatchAllTag = 0;

  unsigned PointerTagShift;
  uint64_t TagMaskByte;

  uint8_t maxShadeSize = 15;

  Function *HwasanCtorFunction;

  FunctionCallee HwasanMemoryAccessCallback[2][kNumberOfAccessSizes][2];
  FunctionCallee HwasanMemoryAccessCallbackDbg[2][kNumberOfAccessSizes][2];
  FunctionCallee HwasanMemoryAccessCallbackSized[2][2];
  FunctionCallee HwasanMemoryAccessCallbackSizedDbg[2][2];

  FunctionCallee HwasanTagMemoryFunc;
  FunctionCallee HwasanTestFreedFunc;
  FunctionCallee HwasanShadeMemoryFunc;
  FunctionCallee HwasanGenerateTagFunc;

  Constant *ShadowGlobal;

  Value *ShadowBase = nullptr;
  Value *StackBaseTag = nullptr;
  GlobalValue *ThreadPtrGlobal = nullptr;
};

class HWAddressSanitizerLegacyPass : public FunctionPass {
public:
  // Pass identification, replacement for typeid.
  static char ID;

  explicit HWAddressSanitizerLegacyPass(bool CompileKernel = false,
                                        bool Recover = false,
                                        bool DisableOptimization = false)
      : FunctionPass(ID), CompileKernel(CompileKernel), Recover(Recover),
        DisableOptimization(DisableOptimization) {
    initializeHWAddressSanitizerLegacyPassPass(
        *PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "HWAddressSanitizer"; }

  bool doInitialization(Module &M) override {
    HWASan = std::make_unique<HWAddressSanitizer>(M, CompileKernel, Recover,
                                                  /*SSI=*/nullptr);
    return true;
  }

  bool runOnFunction(Function &F) override {

    auto TargetTriple = Triple(F.getParent()->getTargetTriple());
    if (shouldUseStackSafetyAnalysis(TargetTriple, DisableOptimization)) {
      // We cannot call getAnalysis in doInitialization, that would cause a
      // crash as the required analyses are not initialized yet.
      HWASan->setSSI(
          &getAnalysis<StackSafetyGlobalInfoWrapperPass>().getResult());
    }
    bool ret = HWASan->sanitizeFunction(
        F,
        [&]() -> const DominatorTree & {
          return getAnalysis<DominatorTreeWrapperPass>().getDomTree();
        },
        [&]() -> const PostDominatorTree & {
          return getAnalysis<PostDominatorTreeWrapperPass>().getPostDomTree();
        });
    return ret;
  }

  bool doFinalization(Module &M) override {
    HWASan.reset();
    return false;
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // This is an over-estimation of, in case we are building for an
    // architecture that doesn't allow stack tagging we will still load the
    // analysis.
    // This is so we don't need to plumb TargetTriple all the way to here.
    if (mightUseStackSafetyAnalysis(DisableOptimization))
      AU.addRequired<StackSafetyGlobalInfoWrapperPass>();
    AU.addRequired<DominatorTreeWrapperPass>();
    AU.addRequired<PostDominatorTreeWrapperPass>();
  }

private:
  std::unique_ptr<HWAddressSanitizer> HWASan;
  bool CompileKernel;
  bool Recover;
  bool DisableOptimization;
};

} // end anonymous namespace

char HWAddressSanitizerLegacyPass::ID = 0;

INITIALIZE_PASS_BEGIN(
    HWAddressSanitizerLegacyPass, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false,
    false)
INITIALIZE_PASS_DEPENDENCY(StackSafetyGlobalInfoWrapperPass)
INITIALIZE_PASS_DEPENDENCY(DominatorTreeWrapperPass)
INITIALIZE_PASS_DEPENDENCY(PostDominatorTreeWrapperPass)
INITIALIZE_PASS_END(
    HWAddressSanitizerLegacyPass, "hwasan",
    "HWAddressSanitizer: detect memory bugs using tagged addressing.", false,
    false)

FunctionPass *
llvm::createHWAddressSanitizerLegacyPassPass(bool CompileKernel, bool Recover,
                                             bool DisableOptimization) {
  assert(!CompileKernel || Recover);
  return new HWAddressSanitizerLegacyPass(CompileKernel, Recover,
                                          DisableOptimization);
}

PreservedAnalyses HWAddressSanitizerPass::run(Module &M,
                                              ModuleAnalysisManager &MAM) {
  const StackSafetyGlobalInfo *SSI = nullptr;
  auto TargetTriple = llvm::Triple(M.getTargetTriple());
  if (shouldUseStackSafetyAnalysis(TargetTriple, Options.DisableOptimization))
    SSI = &MAM.getResult<StackSafetyGlobalAnalysis>(M);

  HWAddressSanitizer HWASan(M, Options.CompileKernel, Options.Recover, SSI);
  bool Modified = false;
  auto &FAM = MAM.getResult<FunctionAnalysisManagerModuleProxy>(M).getManager();
  bool MainInited = false;

  for (Function &F : M) {
    Modified |= HWASan.sanitizeFunction(
        F,
        [&]() -> const DominatorTree & {
          return FAM.getResult<DominatorTreeAnalysis>(F);
        },
        [&]() -> const PostDominatorTree & {
          return FAM.getResult<PostDominatorTreeAnalysis>(F);
        });

    StringRef Name = GlobalValue::dropLLVMManglingEscape(F.getName());

#ifndef DISABLE_SHADE_GLOBALS
    if ((Name == "main" && !MainInited)) {
      for (auto &BB : F) {
        auto I = BB.getFirstNonPHI();
        IRBuilder<> IRB(I);
        for (GlobalVariable *GV : HWASan.StructGlobals) {
          if (StructType *GVType = dyn_cast<StructType>(GV->getValueType())) {
            HWASan.applyShading(IRB, GVType, GV);
          }
        }
        MainInited = true;
        break;
      }
    }
#endif
  }

  if (Modified)
    return PreservedAnalyses::none();
  return PreservedAnalyses::all();
}
void HWAddressSanitizerPass::printPipeline(
    raw_ostream &OS, function_ref<StringRef(StringRef)> MapClassName2PassName) {
  static_cast<PassInfoMixin<HWAddressSanitizerPass> *>(this)->printPipeline(
      OS, MapClassName2PassName);
  OS << "<";
  if (Options.CompileKernel)
    OS << "kernel;";
  if (Options.Recover)
    OS << "recover";
  OS << ">";
}

void HWAddressSanitizer::createHwasanCtorComdat() {
  std::tie(HwasanCtorFunction, std::ignore) =
      getOrCreateSanitizerCtorAndInitFunctions(
          M, kHwasanModuleCtorName, kHwasanInitName,
          /*InitArgTypes=*/{},
          /*InitArgs=*/{},
          // This callback is invoked when the functions are created the first
          // time. Hook them into the global ctors list in that case:
          [&](Function *Ctor, FunctionCallee) {
            Comdat *CtorComdat = M.getOrInsertComdat(kHwasanModuleCtorName);
            Ctor->setComdat(CtorComdat);
            appendToGlobalCtors(M, Ctor, 0, Ctor);
          });

  // Create a note that contains pointers to the list of global
  // descriptors. Adding a note to the output file will cause the linker to
  // create a PT_NOTE program header pointing to the note that we can use to
  // find the descriptor list starting from the program headers. A function
  // provided by the runtime initializes the shadow memory for the globals by
  // accessing the descriptor list via the note. The dynamic loader needs to
  // call this function whenever a library is loaded.
  //
  // The reason why we use a note for this instead of a more conventional
  // approach of having a global constructor pass a descriptor list pointer to
  // the runtime is because of an order of initialization problem. With
  // constructors we can encounter the following problematic scenario:
  //
  // 1) library A depends on library B and also interposes one of B's symbols
  // 2) B's constructors are called before A's (as required for correctness)
  // 3) during construction, B accesses one of its "own" globals (actually
  //    interposed by A) and triggers a HWASAN failure due to the initialization
  //    for A not having happened yet
  //
  // Even without interposition it is possible to run into similar situations in
  // cases where two libraries mutually depend on each other.
  //
  // We only need one note per binary, so put everything for the note in a
  // comdat. This needs to be a comdat with an .init_array section to prevent
  // newer versions of lld from discarding the note.
  //
  // Create the note even if we aren't instrumenting globals. This ensures that
  // binaries linked from object files with both instrumented and
  // non-instrumented globals will end up with a note, even if a comdat from an
  // object file with non-instrumented globals is selected. The note is harmless
  // if the runtime doesn't support it, since it will just be ignored.
  Comdat *NoteComdat = M.getOrInsertComdat(kHwasanModuleCtorName);

  Type *Int8Arr0Ty = ArrayType::get(Int8Ty, 0);
  auto Start =
      new GlobalVariable(M, Int8Arr0Ty, true, GlobalVariable::ExternalLinkage,
                         nullptr, "__start_hwasan_globals");
  Start->setVisibility(GlobalValue::HiddenVisibility);
  Start->setDSOLocal(true);
  auto Stop =
      new GlobalVariable(M, Int8Arr0Ty, true, GlobalVariable::ExternalLinkage,
                         nullptr, "__stop_hwasan_globals");
  Stop->setVisibility(GlobalValue::HiddenVisibility);
  Stop->setDSOLocal(true);

  // Null-terminated so actually 8 bytes, which are required in order to align
  // the note properly.
  auto *Name = ConstantDataArray::get(*C, "LLVM\0\0\0");

  auto *NoteTy = StructType::get(Int32Ty, Int32Ty, Int32Ty, Name->getType(),
                                 Int32Ty, Int32Ty);
  auto *Note =
      new GlobalVariable(M, NoteTy, /*isConstant=*/true,
                         GlobalValue::PrivateLinkage, nullptr, kHwasanNoteName);
  Note->setSection(".note.hwasan.globals");
  Note->setComdat(NoteComdat);
  Note->setAlignment(Align(4));
  Note->setDSOLocal(true);

  // The pointers in the note need to be relative so that the note ends up being
  // placed in rodata, which is the standard location for notes.
  auto CreateRelPtr = [&](Constant *Ptr) {
    return ConstantExpr::getTrunc(
        ConstantExpr::getSub(ConstantExpr::getPtrToInt(Ptr, Int64Ty),
                             ConstantExpr::getPtrToInt(Note, Int64Ty)),
        Int32Ty);
  };
  Note->setInitializer(ConstantStruct::getAnon(
      {ConstantInt::get(Int32Ty, 8),                           // n_namesz
       ConstantInt::get(Int32Ty, 8),                           // n_descsz
       ConstantInt::get(Int32Ty, ELF::NT_LLVM_HWASAN_GLOBALS), // n_type
       Name, CreateRelPtr(Start), CreateRelPtr(Stop)}));
  appendToCompilerUsed(M, Note);

  // Create a zero-length global in hwasan_globals so that the linker will
  // always create start and stop symbols.
  auto Dummy = new GlobalVariable(
      M, Int8Arr0Ty, /*isConstantGlobal*/ true, GlobalVariable::PrivateLinkage,
      Constant::getNullValue(Int8Arr0Ty), "hwasan.dummy.global");
  Dummy->setSection("hwasan_globals");
  Dummy->setComdat(NoteComdat);
  Dummy->setMetadata(LLVMContext::MD_associated,
                     MDNode::get(*C, ValueAsMetadata::get(Note)));
  appendToCompilerUsed(M, Dummy);
}

/// Module-level initialization.
///
/// inserts a call to __hwasan_init to the module's constructor list.
void HWAddressSanitizer::initializeModule() {

  TargetTriple = Triple(M.getTargetTriple());

  // x86_64 currently has two modes:
  // - Intel LAM (default)
  // - pointer aliasing (heap only)
  bool IsX86_64 = TargetTriple.getArch() == Triple::x86_64;
  UsePageAliases = shouldUsePageAliases(TargetTriple);
  // InstrumentWithCalls = shouldInstrumentWithCalls(TargetTriple);
  InstrumentWithCalls = 1;
  InstrumentStack = shouldInstrumentStack(TargetTriple);
#ifdef DISABLE_STACK_INSTRUMENTATION
  InstrumentStack = false;
#endif

  DetectUseAfterScope = shouldDetectUseAfterScope(TargetTriple);
  PointerTagShift = IsX86_64 ? 57 : 56;
  TagMaskByte = IsX86_64 ? 0x3F : 0xFF;

  ClGenerateTagsWithCalls = true;
  Mapping.init(TargetTriple, InstrumentWithCalls);

  C = &(M.getContext());
  IRBuilder<> IRB(*C);
  IntptrTy = IRB.getIntPtrTy(*DL);
  Int8PtrTy = IRB.getInt8PtrTy();
  Int8Ty = IRB.getInt8Ty();
  Int32Ty = IRB.getInt32Ty();

  HwasanCtorFunction = nullptr;

  // Older versions of Android do not have the required runtime support for
  // short granules, global or personality function instrumentation. On other
  // platforms we currently require using the latest version of the runtime.
  bool NewRuntime =
      !TargetTriple.isAndroid() || !TargetTriple.isAndroidVersionLT(30);

  UseShortGranules =
      ClUseShortGranules.getNumOccurrences() ? ClUseShortGranules : NewRuntime;
  OutlinedChecks =
      TargetTriple.isAArch64() && TargetTriple.isOSBinFormatELF() &&
      (ClInlineAllChecks.getNumOccurrences() ? !ClInlineAllChecks : !Recover);

  if (ClMatchAllTag.getNumOccurrences()) {
    if (ClMatchAllTag != -1) {
      HasMatchAllTag = true;
      MatchAllTag = ClMatchAllTag & 0xFF;
    }
  } else if (CompileKernel) {
    HasMatchAllTag = true;
    MatchAllTag = 0xFF;
  }

  // If we don't have personality function support, fall back to landing pads.
  InstrumentLandingPads = ClInstrumentLandingPads.getNumOccurrences()
                              ? ClInstrumentLandingPads
                              : !NewRuntime;

  if (!CompileKernel) {
    createHwasanCtorComdat();
    bool InstrumentGlobals =
        ClGlobals.getNumOccurrences() ? ClGlobals : NewRuntime;

    if (InstrumentGlobals && !UsePageAliases)
      instrumentGlobals();

    bool InstrumentPersonalityFunctions =
        ClInstrumentPersonalityFunctions.getNumOccurrences()
            ? ClInstrumentPersonalityFunctions
            : NewRuntime;
    if (InstrumentPersonalityFunctions)
      instrumentPersonalityFunctions();
  }

  if (!TargetTriple.isAndroid()) {
    Constant *C = M.getOrInsertGlobal("__hwasan_tls", IntptrTy, [&] {
      auto *GV = new GlobalVariable(M, IntptrTy, /*isConstant=*/false,
                                    GlobalValue::ExternalLinkage, nullptr,
                                    "__hwasan_tls", nullptr,
                                    GlobalVariable::InitialExecTLSModel);
      appendToCompilerUsed(M, GV);
      return GV;
    });
    ThreadPtrGlobal = cast<GlobalVariable>(C);
  }

  initializeFunctionDefinitionList(M);
}

void HWAddressSanitizer::initializeCallbacks(Module &M) {
  IRBuilder<> IRB(*C);
  for (size_t AccessIsWrite = 0; AccessIsWrite <= 1; AccessIsWrite++) {
    for (size_t AccessNeedsShade = 0; AccessNeedsShade <= 1;
         AccessNeedsShade++) {
      const std::string TypeStr = AccessIsWrite ? "store" : "load";
      const std::string ShadeStr = AccessNeedsShade ? "_shade" : "";
      const std::string EndingStr = Recover ? "_noabort" : "";

      HwasanMemoryAccessCallbackSizedDbg[AccessIsWrite][AccessNeedsShade] =
          M.getOrInsertFunction(ClMemoryAccessCallbackPrefix + TypeStr + "N" +
                                    EndingStr + ShadeStr + "_dbg",
                                FunctionType::get(IRB.getVoidTy(),
                                                  {IntptrTy, IntptrTy, Int64Ty},
                                                  false));
      for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
           AccessSizeIndex++) {
        HwasanMemoryAccessCallbackDbg[AccessIsWrite][AccessSizeIndex]
                                     [AccessNeedsShade] = M.getOrInsertFunction(
                                         ClMemoryAccessCallbackPrefix +
                                             TypeStr +
                                             itostr(1ULL << AccessSizeIndex) +
                                             EndingStr + ShadeStr + "_dbg",
                                         FunctionType::get(IRB.getVoidTy(),
                                                           {IntptrTy, Int64Ty},
                                                           false));
      }
      HwasanMemoryAccessCallbackSized[AccessIsWrite][AccessNeedsShade] =
          M.getOrInsertFunction(
              ClMemoryAccessCallbackPrefix + TypeStr + "N" + EndingStr +
                  ShadeStr,
              FunctionType::get(IRB.getVoidTy(), {IntptrTy, IntptrTy}, false));

      for (size_t AccessSizeIndex = 0; AccessSizeIndex < kNumberOfAccessSizes;
           AccessSizeIndex++) {
        HwasanMemoryAccessCallback[AccessIsWrite][AccessSizeIndex]
                                  [AccessNeedsShade] = M.getOrInsertFunction(
                                      ClMemoryAccessCallbackPrefix + TypeStr +
                                          itostr(1ULL << AccessSizeIndex) +
                                          EndingStr + ShadeStr,
                                      FunctionType::get(IRB.getVoidTy(),
                                                        {IntptrTy}, false));
      }
    }
  }
  HwasanTestFreedFunc =
      M.getOrInsertFunction("__hwasan_test_free", IRB.getVoidTy(), Int8PtrTy);
  HwasanShadeMemoryFunc = M.getOrInsertFunction(
      "__hwasan_shade_memory", IRB.getVoidTy(), Int8PtrTy, Int8Ty, IntptrTy);
  HwasanTagMemoryFunc =
      M.getOrInsertFunction("__hwasan_inline_tag_memory", IRB.getVoidTy(),
                            Int8PtrTy, Int8Ty, IntptrTy);
  HwasanGenerateTagFunc =
      M.getOrInsertFunction("__hwasan_generate_tag", Int8Ty);

  ShadowGlobal = M.getOrInsertGlobal("__hwasan_shadow",
                                     ArrayType::get(IRB.getInt8Ty(), 0));

  const std::string MemIntrinCallbackPrefix =
      CompileKernel ? std::string("") : ClMemoryAccessCallbackPrefix;
  HWAsanMemmove = M.getOrInsertFunction(
      MemIntrinCallbackPrefix + "memmove", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy, Int8Ty, Int8Ty);
  HWAsanMemcpy = M.getOrInsertFunction(
      MemIntrinCallbackPrefix + "memcpy", IRB.getInt8PtrTy(),
      IRB.getInt8PtrTy(), IRB.getInt8PtrTy(), IntptrTy, Int8Ty, Int8Ty);
  HWAsanMemset = M.getOrInsertFunction(MemIntrinCallbackPrefix + "memset",
                                       IRB.getInt8PtrTy(), IRB.getInt8PtrTy(),
                                       IRB.getInt32Ty(), IntptrTy, Int8Ty);

  HWAsanHandleVfork =
      M.getOrInsertFunction("__hwasan_handle_vfork", IRB.getVoidTy(), IntptrTy);
}

Value *HWAddressSanitizer::getOpaqueNoopCast(IRBuilder<> &IRB, Value *Val) {
  // An empty inline asm with input reg == output reg.
  // An opaque no-op cast, basically.
  // This prevents code bloat as a result of rematerializing trivial definitions
  // such as constants or global addresses at every load and store.
  InlineAsm *Asm =
      InlineAsm::get(FunctionType::get(Int8PtrTy, {Val->getType()}, false),
                     StringRef(""), StringRef("=r,0"),
                     /*hasSideEffects=*/false);
  return IRB.CreateCall(Asm, {Val}, ".hwasan.shadow");
}

Value *HWAddressSanitizer::getDynamicShadowIfunc(IRBuilder<> &IRB) {
  return getOpaqueNoopCast(IRB, ShadowGlobal);
}

Value *HWAddressSanitizer::getShadowNonTls(IRBuilder<> &IRB) {
  if (Mapping.Offset != kDynamicShadowSentinel)
    return getOpaqueNoopCast(
        IRB, ConstantExpr::getIntToPtr(
                 ConstantInt::get(IntptrTy, Mapping.Offset), Int8PtrTy));

  if (Mapping.InGlobal) {
    return getDynamicShadowIfunc(IRB);
  } else {
    Value *GlobalDynamicAddress =
        IRB.GetInsertBlock()->getParent()->getParent()->getOrInsertGlobal(
            kHwasanShadowMemoryDynamicAddress, Int8PtrTy);
    return IRB.CreateLoad(Int8PtrTy, GlobalDynamicAddress);
  }
}

bool HWAddressSanitizer::ignoreAccess(Instruction *Inst, Value *Ptr) {
  // Do not instrument acesses from different address spaces; we cannot deal
  // with them.
  Type *PtrTy = cast<PointerType>(Ptr->getType()->getScalarType());
  if (PtrTy->getPointerAddressSpace() != 0)
    return true;

  // Ignore swifterror addresses.
  // swifterror memory addresses are mem2reg promoted by instruction
  // selection. As such they cannot have regular uses like an instrumentation
  // function and it makes no sense to track them as memory.
  if (Ptr->isSwiftError())
    return true;

  if (findAllocaForValue(Ptr)) {
    if (!InstrumentStack)
      return true;
    if (SSI && SSI->stackAccessIsSafe(*Inst))
      return true;
  }
  return false;
}

void HWAddressSanitizer::getInterestingMemoryOperands(
    Instruction *I, SmallVectorImpl<InterestingMemoryOperand> &Interesting) {
  // Skip memory accesses inserted by another instrumentation.
  if (I->hasMetadata("nosanitize"))
    return;

  // Do not instrument the load fetching the dynamic shadow address.
  if (ShadowBase == I)
    return;

  if (LoadInst *LI = dyn_cast<LoadInst>(I)) {
    if (!ClInstrumentReads || ignoreAccess(I, LI->getPointerOperand()))
      return;
    Interesting.emplace_back(I, LI->getPointerOperandIndex(), false,
                             LI->getType(), LI->getAlign());
  } else if (StoreInst *SI = dyn_cast<StoreInst>(I)) {
    if (!ClInstrumentWrites || ignoreAccess(I, SI->getPointerOperand()))
      return;
    Interesting.emplace_back(I, SI->getPointerOperandIndex(), true,
                             SI->getValueOperand()->getType(), SI->getAlign());
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!ClInstrumentAtomics || ignoreAccess(I, RMW->getPointerOperand()))
      return;
    Interesting.emplace_back(I, RMW->getPointerOperandIndex(), true,
                             RMW->getValOperand()->getType(), None);
  } else if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I)) {
    if (!ClInstrumentAtomics || ignoreAccess(I, XCHG->getPointerOperand()))
      return;
    Interesting.emplace_back(I, XCHG->getPointerOperandIndex(), true,
                             XCHG->getCompareOperand()->getType(), None);
  } else if (auto CI = dyn_cast<CallInst>(I)) {
    for (unsigned ArgNo = 0; ArgNo < CI->arg_size(); ArgNo++) {
      if (!ClInstrumentByval || !CI->isByValArgument(ArgNo) ||
          ignoreAccess(I, CI->getArgOperand(ArgNo)))
        continue;
      Type *Ty = CI->getParamByValType(ArgNo);
      Interesting.emplace_back(I, ArgNo, false, Ty, Align(1));
    }
  }
}

static unsigned getPointerOperandIndex(Instruction *I) {
  if (LoadInst *LI = dyn_cast<LoadInst>(I))
    return LI->getPointerOperandIndex();
  if (StoreInst *SI = dyn_cast<StoreInst>(I))
    return SI->getPointerOperandIndex();
  if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I))
    return RMW->getPointerOperandIndex();
  if (AtomicCmpXchgInst *XCHG = dyn_cast<AtomicCmpXchgInst>(I))
    return XCHG->getPointerOperandIndex();
  report_fatal_error("Unexpected instruction");
  return -1;
}

static size_t TypeSizeToSizeIndex(uint32_t TypeSize) {
  size_t Res = countTrailingZeros(TypeSize / 8);
  assert(Res < kNumberOfAccessSizes);
  return Res;
}

bool HWAddressSanitizer::isAllocationWrapper(CallBase *CB) {
#if FIND_ALLOC_WRAPPERS
  if (CB->getCalledFunction() == nullptr)
    return false;

  if (Function *F = dyn_cast<Function>(CB->getCalledFunction())) {
    if (F->getInstructionCount() == 0)
      return false;
    if (ReturnInst *I = dyn_cast<ReturnInst>(&F->back().back())) {
      if (I->getReturnValue() == nullptr)
        return false;
      if (CallBase *AF = dyn_cast<CallBase>(I->getReturnValue())) {
        if (isAllocationFunction(AF)) {
          return true;
        }
      }
    }
  }
#endif

  return false;
}

bool HWAddressSanitizer::isAllocationFunction(CallBase *CB) {
  auto allocation_names = {
      "malloc", "calloc",
      "realloc"}; //, "_Znam", "_Znwm", "_ZnamRKSt9nothrow_t",
                  //"_ZnwmRKSt9nothrow_t"};

  if (CB->getCalledFunction()) {
    if (CB->getCalledFunction()->hasName()) {
      if (std::find(allocation_names.begin(), allocation_names.end(),
                    CB->getCalledFunction()->getName()) !=
          allocation_names.end()) {
        return true;
      }
    }
  }
  return false;
}

bool HWAddressSanitizer::isNotStruct(Type *T) {
  if (T->isPointerTy())
    return isNotStruct(T->getPointerElementType());
  if (T->isArrayTy())
    return isNotStruct(T->getArrayElementType());
  return T->isSingleValueType();
}

// returns true iff I is definitely not part of a struct
bool HWAddressSanitizer::isNotPartOfStruct(Value *I) {
  std::queue<Value *> Q;
  Q.push(I);
  int cost = 0;

  while (!Q.empty()) {
    if (cost++ > 100)
      return false;

    I = Q.front();
    Q.pop();
    if (isa<GEPOperator, BitCastOperator, LoadInst, IntToPtrInst, PtrToIntInst>(
            I)) {
      Q.push(dyn_cast<User>(I)->getOperand(0));
    } else if (PHINode *PN = dyn_cast<PHINode>(I)) {
      for (Value *V : PN->incoming_values()) {
        Q.push(V);
      }
    } else if (Argument *A = dyn_cast<Argument>(I)) {
      Function *AP = A->getParent();
      if (AP->hasAddressTaken() || AP->isVarArg())
        return false;
      for (User *U : AP->users()) {
        Q.push(dyn_cast<CallBase>(U)->getArgOperand(A->getArgNo()));
      }
    } else if (BinaryOperator *BC = dyn_cast<BinaryOperator>(I)) {
      Q.push(BC->getOperand(0));
      Q.push(BC->getOperand(1));
    } else if (AllocaInst *AI = dyn_cast<AllocaInst>(I)) {
      if (!isNotStruct(AI->getAllocatedType()))
        return false;
    } else if (CallBase *CB = dyn_cast<CallBase>(I)) {
      if (!isAllocationFunction(CB))
        return false;
      Instruction *NI = CB->getNextNonDebugInstruction();
      if (!isa<BitCastOperator>(NI))
        return false;
      if (!isNotStruct(NI->getType()))
        return false;
    } else if (isa<GlobalValue, Constant>(I)) {
      if (!isNotStruct(I->getType()))
        return false;
    } else {
      return false;
    }
  }
  return true;
}

void HWAddressSanitizer::untagPointerOperand(Instruction *I, Value *Addr) {
  if (TargetTriple.isAArch64() || TargetTriple.getArch() == Triple::x86_64)
    return;

  IRBuilder<> IRB(I);
  Value *AddrLong = IRB.CreatePointerCast(Addr, IntptrTy);
  Value *UntaggedPtr =
      IRB.CreateIntToPtr(untagPointer(IRB, AddrLong), Addr->getType());
  I->setOperand(getPointerOperandIndex(I), UntaggedPtr);
}

Value *HWAddressSanitizer::memToShadow(Value *Mem, IRBuilder<> &IRB) {
  return IRB.CreateXor(
      Mem, llvm::ConstantInt::get(Int64Ty, 0x400000000000ULL, false));
}

Value *HWAddressSanitizer::memToShade(Value *Mem, IRBuilder<> &IRB) {
  return IRB.CreateXor(
      Mem, llvm::ConstantInt::get(Int64Ty, 0x400000000000ULL, false));
}

int64_t HWAddressSanitizer::getAccessInfo(bool IsWrite,
                                          unsigned AccessSizeIndex) {
  return (CompileKernel << HWASanAccessInfo::CompileKernelShift) +
         (HasMatchAllTag << HWASanAccessInfo::HasMatchAllShift) +
         (MatchAllTag << HWASanAccessInfo::MatchAllShift) +
         (Recover << HWASanAccessInfo::RecoverShift) +
         (IsWrite << HWASanAccessInfo::IsWriteShift) +
         (AccessSizeIndex << HWASanAccessInfo::AccessSizeShift);
}

void HWAddressSanitizer::instrumentMemAccessOutline(Value *Ptr, bool IsWrite,
                                                    unsigned AccessSizeIndex,
                                                    Instruction *InsertBefore) {
  assert(!UsePageAliases);
  const int64_t AccessInfo = getAccessInfo(IsWrite, AccessSizeIndex);
  IRBuilder<> IRB(InsertBefore);
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  Ptr = IRB.CreateBitCast(Ptr, Int8PtrTy);
  IRB.CreateCall(Intrinsic::getDeclaration(
                     M, UseShortGranules
                            ? Intrinsic::hwasan_check_memaccess_shortgranules
                            : Intrinsic::hwasan_check_memaccess),
                 {ShadowBase, Ptr, ConstantInt::get(Int32Ty, AccessInfo)});
}

void HWAddressSanitizer::instrumentMemAccessInline(Value *Ptr, bool IsWrite,
                                                   unsigned AccessSizeIndex,
                                                   Instruction *InsertBefore) {
  assert(!UsePageAliases);
  const int64_t AccessInfo = getAccessInfo(IsWrite, AccessSizeIndex);
  IRBuilder<> IRB(InsertBefore);

  Value *PtrLong = IRB.CreatePointerCast(Ptr, IntptrTy);
  Value *PtrTag = IRB.CreateTrunc(IRB.CreateLShr(PtrLong, PointerTagShift),
                                  IRB.getInt8Ty());
  Value *AddrLong = untagPointer(IRB, PtrLong);
  Value *Shadow = memToShadow(AddrLong, IRB);
  Value *MemTag = IRB.CreateLoad(Int8Ty, Shadow);
  Value *TagMismatch = IRB.CreateICmpNE(PtrTag, MemTag);

  if (HasMatchAllTag) {
    Value *TagNotIgnored = IRB.CreateICmpNE(
        PtrTag, ConstantInt::get(PtrTag->getType(), MatchAllTag));
    TagMismatch = IRB.CreateAnd(TagMismatch, TagNotIgnored);
  }

  Instruction *CheckTerm =
      SplitBlockAndInsertIfThen(TagMismatch, InsertBefore, false,
                                MDBuilder(*C).createBranchWeights(1, 100000));

  IRB.SetInsertPoint(CheckTerm);
  Value *OutOfShortGranuleTagRange =
      IRB.CreateICmpUGT(MemTag, ConstantInt::get(Int8Ty, 15));
  Instruction *CheckFailTerm =
      SplitBlockAndInsertIfThen(OutOfShortGranuleTagRange, CheckTerm, !Recover,
                                MDBuilder(*C).createBranchWeights(1, 100000));

  IRB.SetInsertPoint(CheckTerm);
  Value *PtrLowBits = IRB.CreateTrunc(IRB.CreateAnd(PtrLong, 15), Int8Ty);
  PtrLowBits = IRB.CreateAdd(
      PtrLowBits, ConstantInt::get(Int8Ty, (1 << AccessSizeIndex) - 1));
  Value *PtrLowBitsOOB = IRB.CreateICmpUGE(PtrLowBits, MemTag);
  SplitBlockAndInsertIfThen(PtrLowBitsOOB, CheckTerm, false,
                            MDBuilder(*C).createBranchWeights(1, 100000),
                            (DomTreeUpdater *)nullptr, nullptr,
                            CheckFailTerm->getParent());

  IRB.SetInsertPoint(CheckTerm);
  Value *InlineTagAddr = IRB.CreateOr(AddrLong, 15);
  InlineTagAddr = IRB.CreateIntToPtr(InlineTagAddr, Int8PtrTy);
  Value *InlineTag = IRB.CreateLoad(Int8Ty, InlineTagAddr);
  Value *InlineTagMismatch = IRB.CreateICmpNE(PtrTag, InlineTag);
  SplitBlockAndInsertIfThen(InlineTagMismatch, CheckTerm, false,
                            MDBuilder(*C).createBranchWeights(1, 100000),
                            (DomTreeUpdater *)nullptr, nullptr,
                            CheckFailTerm->getParent());

  IRB.SetInsertPoint(CheckFailTerm);
  InlineAsm *Asm;
  switch (TargetTriple.getArch()) {
  case Triple::x86_64:
    // The signal handler will find the data address in rdi.
    Asm = InlineAsm::get(
        FunctionType::get(IRB.getVoidTy(), {PtrLong->getType()}, false),
        "int3\nnopl " +
            itostr(0x40 + (AccessInfo & HWASanAccessInfo::RuntimeMask)) +
            "(%rax)",
        "{rdi}",
        /*hasSideEffects=*/true);
    break;
  case Triple::aarch64:
  case Triple::aarch64_be:
    // The signal handler will find the data address in x0.
    Asm = InlineAsm::get(
        FunctionType::get(IRB.getVoidTy(), {PtrLong->getType()}, false),
        "brk #" + itostr(0x900 + (AccessInfo & HWASanAccessInfo::RuntimeMask)),
        "{x0}",
        /*hasSideEffects=*/true);
    break;
  default:
    report_fatal_error("unsupported architecture");
  }
  IRB.CreateCall(Asm, PtrLong);
  if (Recover)
    cast<BranchInst>(CheckFailTerm)->setSuccessor(0, CheckTerm->getParent());
}

bool HWAddressSanitizer::ignoreMemIntrinsic(MemIntrinsic *MI) {
  if (MemTransferInst *MTI = dyn_cast<MemTransferInst>(MI)) {
    return (!ClInstrumentWrites || ignoreAccess(MTI, MTI->getDest())) &&
           (!ClInstrumentReads || ignoreAccess(MTI, MTI->getSource()));
  }
  if (isa<MemSetInst>(MI))
    return !ClInstrumentWrites || ignoreAccess(MI, MI->getDest());
  return false;
}

void HWAddressSanitizer::instrumentMemIntrinsic(MemIntrinsic *MI) {
  IRBuilder<> IRB(MI);
  if (isa<MemTransferInst>(MI)) {
    IRB.CreateCall(
        isa<MemMoveInst>(MI) ? HWAsanMemmove : HWAsanMemcpy,
        {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
         IRB.CreatePointerCast(MI->getOperand(1), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false),
         ConstantInt::get(Int8Ty, 0), ConstantInt::get(Int8Ty, 0)});
  } else if (isa<MemSetInst>(MI)) {
    IRB.CreateCall(
        HWAsanMemset,
        {IRB.CreatePointerCast(MI->getOperand(0), IRB.getInt8PtrTy()),
         IRB.CreateIntCast(MI->getOperand(1), IRB.getInt32Ty(), false),
         IRB.CreateIntCast(MI->getOperand(2), IntptrTy, false),
         ConstantInt::get(Int8Ty, 0)});
  }
  MI->eraseFromParent();
}

bool HWAddressSanitizer::instrumentCMP(Instruction *CMP) {
  auto op1 = CMP->getOperand(0);
  auto cmpType = op1->getType();
  auto op2 = CMP->getOperand(1);

  if (cmpType->isPointerTy()) {
    IRBuilder<> IRB(CMP);
    auto untaggedop1 = untagPointer(IRB, IRB.CreatePtrToInt(op1, IntptrTy));
    auto untaggedop2 = untagPointer(IRB, IRB.CreatePtrToInt(op2, IntptrTy));
    CMP->replaceUsesOfWith(op1, IRB.CreateIntToPtr(untaggedop1, cmpType));
    CMP->replaceUsesOfWith(op2, IRB.CreateIntToPtr(untaggedop2, cmpType));
  }

  return false;
}

bool HWAddressSanitizer::containsUnion(StructType *ST) {

  if (!ST->isLiteral()) {
    if (ST->getName().startswith("union")) {
      return true;
    }
  }

  for (unsigned int i = 0; i < ST->getNumElements(); ++i) {
    llvm::Type *fieldType = ST->getElementType(i);

    if (fieldType->isStructTy()) {
      if (containsUnion(llvm::cast<llvm::StructType>(fieldType))) {
        return true;
      }
    }
  }

  return false;
}

std::vector<std::tuple<int, int>>
HWAddressSanitizer::getMetadataLayoutFromType(StructType *ST) {
  std::vector<std::tuple<int, int>> bufferVec;

  const llvm::StructLayout *Layout = DL->getStructLayout(ST);
  auto nContainedTypes = ST->getNumElements();

  uint8_t shade = 1;
  for (unsigned int i = 0; i < nContainedTypes; i++) {

    unsigned int size;
    if (i == nContainedTypes -
                 1) { // last field's size is total struct size minus its offset
      size = Layout->getSizeInBytes() - Layout->getElementOffset(i);
    } else { // size is difference between next field's offset and this field's
             // offset
      size = Layout->getElementOffset(i + 1) - Layout->getElementOffset(i);
    }

    assert((size > 0) && "zero_size");

    auto elementType = ST->getContainedType(i);
    if (StructType *nestedST = dyn_cast<StructType>(elementType)) {
      auto subBufferVec = getMetadataLayoutFromType(nestedST);

      bufferVec.insert(bufferVec.end(), subBufferVec.begin(),
                       subBufferVec.end());
      uint64_t subBufferVecTotalSize = 0;
      for (unsigned long j = 0; j < subBufferVec.size(); j++) {
        subBufferVecTotalSize += std::get<1>(subBufferVec[j]);
      }
      if (subBufferVecTotalSize < size) {
        // Pad according to size
        unsigned int difference = size - subBufferVecTotalSize;

        bufferVec.push_back(std::make_tuple(0, difference));
      }
    } else {
      if (elementType->isArrayTy()) {
        auto aType = dyn_cast<ArrayType>(elementType);
        auto nType = aType->getElementType();
        
        if (nType->isStructTy()) {
          bufferVec.push_back(std::make_tuple(0, size));
        } else {
          bufferVec.push_back(std::make_tuple(shade, size));
        }
      } else if (elementType->isVectorTy()) {
        auto vType = dyn_cast<VectorType>(elementType);
        bufferVec.push_back(std::make_tuple(shade, size));
        if (vType->getElementType()->isStructTy()) {
          assert(0 && "Vector with StructTypes not yet handeled");
        }
      } else {
        bufferVec.push_back(std::make_tuple(shade, size));
      }
    }
    shade = shade % (maxShadeSize);
    shade += 1;
  }

  return bufferVec;
}

bool HWAddressSanitizer::instrumentGEP(GetElementPtrInst *GEPI) {
  auto SourceType = GEPI->getSourceElementType();
  auto ResultType = GEPI->getResultElementType();
  if (PointerType *BIPtrType = dyn_cast<PointerType>(ResultType)) {
    auto elementType = BIPtrType->getPointerElementType();
    if (auto STType = dyn_cast<StructType>(elementType)) {
      if (containsUnion(STType)) {
        return false;
      }
    }
  }

  IRBuilder<> IRB(GEPI->getNextNode());
  if (SourceType->isStructTy() && (!ResultType->isStructTy())) {


    auto nOperands = GEPI->getNumOperands();

    if (nOperands == 3) {
      auto firstIndex = GEPI->getOperand(1);
      bool positiveFirstIndex = true;
      if (ConstantInt *CI = dyn_cast<ConstantInt>(firstIndex)) {
        if (CI->getBitWidth() <= 64) {
          // if (CI->getSExtValue() < 0) {
          if (CI->getSExtValue() != 0) {
            positiveFirstIndex = false;
          };
        }
      }

      // Get Indexed Element from GEPI, then Shade
      auto secondIndex = GEPI->getOperand(2);
      
      auto index_mod8 = IRB.CreateURem(
          secondIndex, ConstantInt::get(secondIndex->getType(), maxShadeSize));

      bool nonZeroIndex = true;

      auto incremented = IRB.CreateAdd(
          index_mod8, ConstantInt::get(secondIndex->getType(), 1));
      if (nonZeroIndex && positiveFirstIndex) {
        Value *GEPILong = IRB.CreateBitOrPointerCast(GEPI, IntptrTy);
        Value *Replacement =
            reshadePointer(IRB, GEPI->getType(), GEPILong,
                           IRB.CreateZExt(incremented, IntptrTy));
        std::string Name = GEPI->getName().str();
        Replacement->setName(Name + ".shaded");

        GEPI->replaceUsesWithIf(Replacement, [GEPILong](Use &U) {
          return U.getUser() != GEPILong;
        });
      }
    }
  }
  return false;
}

bool HWAddressSanitizer::instrumentConstGEP(ConstantExpr *ConstGEPI) {
  auto SourceType = ConstGEPI->getOperand(0)->getType();
  if (PointerType *BIPtrType = dyn_cast<PointerType>(SourceType)) {
    SourceType = BIPtrType->getPointerElementType();
  }

  auto ResultType = ConstGEPI->getType();
  if (PointerType *BIPtrType = dyn_cast<PointerType>(ResultType)) {
    auto elementType = BIPtrType->getPointerElementType();
    if (auto STType = dyn_cast<StructType>(elementType)) {
      if (!STType->isLiteral()) {
        if (elementType->getStructName().startswith("union")) {
          return false;
        }
      }
    }
  }
  
  if (SourceType->isStructTy() && !ResultType->isStructTy()) {

    auto nOperands = ConstGEPI->getNumOperands();

    if (nOperands == 3) {

      auto firstIndex = ConstGEPI->getOperand(1);
      bool positiveFirstIndex = true;
      if (ConstantInt *CI = dyn_cast<ConstantInt>(firstIndex)) {
        if (CI->getBitWidth() <= 64) {
          
          if (CI->getSExtValue() != 0) {
            positiveFirstIndex = false;
          };
        }
      }

      // Get Indexed Element from GEPI, then Shade
      auto secondIndex = ConstGEPI->getOperand(2);

      auto index_mod8 = llvm::ConstantExpr::getURem(
          secondIndex, ConstantInt::get(secondIndex->getType(), maxShadeSize));
          
      bool nonZeroIndex = true;
      if (ConstantInt *CI = dyn_cast<ConstantInt>(index_mod8)) {
        if (CI->getBitWidth() <= 64) {
          if (CI->getSExtValue() == 0) {
            nonZeroIndex = false;
          };
        }
      }
      auto incremented = llvm::ConstantExpr::getAdd(
          index_mod8, ConstantInt::get(secondIndex->getType(), 1));
      if (nonZeroIndex && positiveFirstIndex) {
        llvm::ConstantInt *CI = dyn_cast<llvm::ConstantInt>(incremented);
        
        if (CI->getSExtValue() > 0) {

          Constant *ConstGEPILong =
              llvm::ConstantExpr::getPointerCast(ConstGEPI, IntptrTy);
          Constant *Replacement = reshadeGlobalPointer(
              ConstGEPI->getType(), ConstGEPILong,
              llvm::ConstantExpr::getZExt(incremented, IntptrTy));
          std::string Name = ConstGEPI->getName().str();
          Replacement->setName(Name + ".shaded");

          ConstGEPI->replaceUsesWithIf(Replacement, [ConstGEPILong](Use &U) {
            auto User = U.getUser();
            if (dyn_cast<Instruction>(User)) {
              if (U.getUser() != ConstGEPILong) {
                return true;
              }
            } else {
              return false;
            }
          });
        }
      }
    }
    return true;
  }
  return false;
}

// For malloc we want to check if the allocated memory is used for a struct.
// For structs we then apply the shade to memory.
void HWAddressSanitizer::instrumentMalloc(CallInst *CI) {
  for (User *user : CI->users()) {
    if (BitCastInst *BI = dyn_cast<BitCastInst>(user)) {
      IRBuilder<> IRB(BI->getNextNode());
      if (PointerType *BIPtrType = dyn_cast<PointerType>(BI->getDestTy())) {
        if (StructType *BIType =
                dyn_cast<StructType>(BIPtrType->getPointerElementType())) {
          applyShading(IRB, BIType, BI);
        }
      }
      return;
    }
  }
}

bool HWAddressSanitizer::instrumentMemAccess(InterestingMemoryOperand &O) {
#ifdef DISABLE_SHADE_CHECKS
  O.shaded = false;
#endif
  static unsigned long check_counter = 0;
  Value *Addr = O.getPtr();

  LLVM_DEBUG(dbgs() << "Instrumenting: " << O.getInsn() << "\n");

  if (O.MaybeMask)
    return false; // FIXME

  IRBuilder<> IRB(O.getInsn());
  if (isPowerOf2_64(O.TypeSize) &&
      (O.TypeSize / 8 <= (1ULL << (kNumberOfAccessSizes - 1))) &&
      (!O.Alignment || *O.Alignment >= (1ULL << Mapping.Scale) ||
       *O.Alignment >= O.TypeSize / 8)) {
    size_t AccessSizeIndex = TypeSizeToSizeIndex(O.TypeSize);
    if (InstrumentWithCalls) {
#ifdef DEBUG_CHECKS
      IRB.CreateCall(
          HwasanMemoryAccessCallbackDbg[O.IsWrite][AccessSizeIndex][O.shaded],
          {IRB.CreatePointerCast(Addr, IntptrTy),
           llvm::ConstantInt::get(Int64Ty, check_counter, false)});
      check_counter++;
#else
      IRB.CreateCall(
          HwasanMemoryAccessCallback[O.IsWrite][AccessSizeIndex][O.shaded],
          {IRB.CreatePointerCast(Addr, IntptrTy)});
#endif
    } else if (OutlinedChecks) {
      instrumentMemAccessOutline(Addr, O.IsWrite, AccessSizeIndex, O.getInsn());
    } else {
      instrumentMemAccessInline(Addr, O.IsWrite, AccessSizeIndex, O.getInsn());
    }
  } else {
#ifdef DEBUG_CHECKS
    IRB.CreateCall(HwasanMemoryAccessCallbackSizedDbg[O.IsWrite][O.shaded],
                   {IRB.CreatePointerCast(Addr, IntptrTy),
                    ConstantInt::get(IntptrTy, O.TypeSize / 8),
                    llvm::ConstantInt::get(Int64Ty, check_counter, false)});
    check_counter++;
#else
    IRB.CreateCall(HwasanMemoryAccessCallbackSized[O.IsWrite][O.shaded],
                   {IRB.CreatePointerCast(Addr, IntptrTy),
                    ConstantInt::get(IntptrTy, O.TypeSize / 8)});
#endif
  }
  untagPointerOperand(O.getInsn(), Addr);

  return true;
}

static uint64_t getAllocaSizeInBytes(const AllocaInst &AI) {
  uint64_t ArraySize = 1;
  if (AI.isArrayAllocation()) {
    const ConstantInt *CI = dyn_cast<ConstantInt>(AI.getArraySize());
    assert(CI && "non-constant array size");
    ArraySize = CI->getZExtValue();
  }
  Type *Ty = AI.getAllocatedType();
  uint64_t SizeInBytes = AI.getModule()->getDataLayout().getTypeAllocSize(Ty);
  return SizeInBytes * ArraySize;
}

void HWAddressSanitizer::shadeAlloca(IRBuilder<> &IRB, AllocaInst *AI,
                                     Value *Tag, size_t Size) {
  size_t AlignedSize = alignTo(Size, Mapping.getObjectAlignment());
  if (!UseShortGranules)
    Size = AlignedSize;

  Value *JustTag = IRB.CreateTrunc(Tag, IRB.getInt8Ty());
  if (InstrumentWithCalls) {
    IRB.CreateCall(HwasanTagMemoryFunc,
                   {IRB.CreatePointerCast(AI, Int8PtrTy), JustTag,
                    ConstantInt::get(IntptrTy, AlignedSize)});
    auto AIType = AI->getAllocatedType();
    if (AIType->isStructTy()) {
      applyShading(IRB, dyn_cast<StructType>(AIType), AI);
    }
  }
}

void HWAddressSanitizer::tagAlloca(IRBuilder<> &IRB, AllocaInst *AI, Value *Tag,
                                   size_t Size) {
  size_t AlignedSize = alignTo(Size, Mapping.getObjectAlignment());
  if (!UseShortGranules)
    Size = AlignedSize;

  Value *JustTag = IRB.CreateTrunc(Tag, IRB.getInt8Ty());
  if (InstrumentWithCalls) {
    IRB.CreateCall(HwasanTagMemoryFunc,
                   {IRB.CreatePointerCast(AI, Int8PtrTy), JustTag,
                    ConstantInt::get(IntptrTy, AlignedSize)});
  } else {
    size_t ShadowSize = Size >> Mapping.Scale;
    Value *ShadowPtr = memToShadow(IRB.CreatePointerCast(AI, IntptrTy), IRB);
    // If this memset is not inlined, it will be intercepted in the hwasan
    // runtime library. That's OK, because the interceptor skips the checks if
    // the address is in the shadow region.
    // FIXME: the interceptor is not as fast as real memset. Consider lowering
    // llvm.memset right here into either a sequence of stores, or a call to
    // hwasan_tag_memory.
    if (ShadowSize)
      IRB.CreateMemSet(ShadowPtr, JustTag, ShadowSize, Align(1));
    if (Size != AlignedSize) {
      IRB.CreateStore(
          ConstantInt::get(Int8Ty, Size % Mapping.getObjectAlignment()),
          IRB.CreateConstGEP1_32(Int8Ty, ShadowPtr, ShadowSize));
      IRB.CreateStore(JustTag, IRB.CreateConstGEP1_32(
                                   Int8Ty, IRB.CreateBitCast(AI, Int8PtrTy),
                                   AlignedSize - 1));
    }
  }
}

unsigned HWAddressSanitizer::retagMask(unsigned AllocaNo) {
  if (TargetTriple.getArch() == Triple::x86_64)
    return AllocaNo & TagMaskByte;

  // A list of 8-bit numbers that have at most one run of non-zero bits.
  // x = x ^ (mask << 56) can be encoded as a single armv8 instruction for these
  // masks.
  // The list does not include the value 255, which is used for UAR.
  //
  // Because we are more likely to use earlier elements of this list than later
  // ones, it is sorted in increasing order of probability of collision with a
  // mask allocated (temporally) nearby. The program that generated this list
  // can be found at:
  // https://github.com/google/sanitizers/blob/master/hwaddress-sanitizer/sort_masks.py
  static unsigned FastMasks[] = {0,  128, 64,  192, 32,  96,  224, 112, 240,
                                 48, 16,  120, 248, 56,  24,  8,   124, 252,
                                 60, 28,  12,  4,   126, 254, 62,  30,  14,
                                 6,  2,   127, 63,  31,  15,  7,   3,   1};
  return FastMasks[AllocaNo % (sizeof(FastMasks) / sizeof(FastMasks[0]))];
}

Value *HWAddressSanitizer::applyTagMask(IRBuilder<> &IRB, Value *OldTag) {
  if (TargetTriple.getArch() == Triple::x86_64) {
    Constant *TagMask = ConstantInt::get(IntptrTy, TagMaskByte);
    Value *NewTag = IRB.CreateAnd(OldTag, TagMask);
    return NewTag;
  }
  // aarch64 uses 8-bit tags, so no mask is needed.
  return OldTag;
}

Value *HWAddressSanitizer::getNextTagWithCall(IRBuilder<> &IRB) {
  return IRB.CreateZExt(IRB.CreateCall(HwasanGenerateTagFunc), IntptrTy);
}

Value *HWAddressSanitizer::getStackBaseTag(IRBuilder<> &IRB) {
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  if (StackBaseTag)
    return StackBaseTag;
  // FIXME: use addressofreturnaddress (but implement it in aarch64 backend
  // first).
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  auto GetStackPointerFn = Intrinsic::getDeclaration(
      M, Intrinsic::frameaddress,
      IRB.getInt8PtrTy(M->getDataLayout().getAllocaAddrSpace()));
  Value *StackPointer = IRB.CreateCall(
      GetStackPointerFn, {Constant::getNullValue(IRB.getInt32Ty())});

  // Extract some entropy from the stack pointer for the tags.
  // Take bits 20..28 (ASLR entropy) and xor with bits 0..8 (these differ
  // between functions).
  Value *StackPointerLong = IRB.CreatePointerCast(StackPointer, IntptrTy);
  Value *StackTag =
      applyTagMask(IRB, IRB.CreateXor(StackPointerLong,
                                      IRB.CreateLShr(StackPointerLong, 20)));
  StackTag->setName("hwasan.stack.base.tag");
  return StackTag;
}

Value *HWAddressSanitizer::getAllocaTag(IRBuilder<> &IRB, Value *StackTag,
                                        AllocaInst *AI, unsigned AllocaNo) {
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  return IRB.CreateXor(StackTag,
                       ConstantInt::get(IntptrTy, retagMask(AllocaNo)));
}

Value *HWAddressSanitizer::getUARTag(IRBuilder<> &IRB, Value *StackTag) {
  if (ClUARRetagToZero)
    return ConstantInt::get(IntptrTy, 0);
  if (ClGenerateTagsWithCalls)
    return getNextTagWithCall(IRB);
  return IRB.CreateXor(StackTag, ConstantInt::get(IntptrTy, TagMaskByte));
}

// Add a tag to an address.
Value *HWAddressSanitizer::tagPointer(IRBuilder<> &IRB, Type *Ty,
                                      Value *PtrLong, Value *Tag) {
  assert(!UsePageAliases);
  Value *TaggedPtrLong;
  if (CompileKernel) {
    // Kernel addresses have 0xFF in the most significant byte.
    Value *ShiftedTag =
        IRB.CreateOr(IRB.CreateShl(Tag, PointerTagShift),
                     ConstantInt::get(IntptrTy, (1ULL << PointerTagShift) - 1));
    TaggedPtrLong = IRB.CreateAnd(PtrLong, ShiftedTag);
  } else {
    // Userspace can simply do OR (tag << PointerTagShift);
    Value *ShiftedTag = IRB.CreateShl(Tag, PointerTagShift);
    TaggedPtrLong = IRB.CreateOr(PtrLong, ShiftedTag);
  }
  return IRB.CreateIntToPtr(TaggedPtrLong, Ty);
}

Value *HWAddressSanitizer::reshadePointer(IRBuilder<> &IRB, Type *Ty,
                                          Value *PtrLong, Value *Shade) {
  // Userspace can simply do OR (tag << PointerTagShift);
  Value *ShiftedTag = IRB.CreateShl(Shade, PointerTagShift);
  auto clearShadeMask = ConstantInt::get(IntptrTy, 0xF0FFFFFFFFFFFFFF);
  auto NonShadedPtrLong = IRB.CreateAnd(PtrLong, clearShadeMask);
  Value *TaggedPtrLong = IRB.CreateOr(NonShadedPtrLong, ShiftedTag);
  return IRB.CreateIntToPtr(TaggedPtrLong, Ty);
}

// Reshade a constant address.
Constant *HWAddressSanitizer::reshadeGlobalPointer(Type *Ty, Constant *PtrLong,
                                                   Constant *Shade) {
  assert(!UsePageAliases);
  Constant *ConstPointerTagShift = ConstantInt::get(IntptrTy, PointerTagShift);
  Constant *ShiftedTag =
      llvm::ConstantExpr::getShl(Shade, ConstPointerTagShift);
  auto clearShadeMask = ConstantInt::get(IntptrTy, 0xF0FFFFFFFFFFFFFF);
  auto NonShadedPtrLong = llvm::ConstantExpr::getAnd(PtrLong, clearShadeMask);
  Constant *TaggedPtrLong =
      llvm::ConstantExpr::getOr(NonShadedPtrLong, ShiftedTag);
  return llvm::ConstantExpr::getIntToPtr(TaggedPtrLong, Ty);
}

// Remove tag from an address.
Value *HWAddressSanitizer::untagPointer(IRBuilder<> &IRB, Value *PtrLong) {
  assert(!UsePageAliases);
  Value *UntaggedPtrLong;
  if (CompileKernel) {
    // Kernel addresses have 0xFF in the most significant byte.
    UntaggedPtrLong =
        IRB.CreateOr(PtrLong, ConstantInt::get(PtrLong->getType(),
                                               0xFFULL << PointerTagShift));
  } else {
    // Userspace addresses have 0x00.
    UntaggedPtrLong =
        IRB.CreateAnd(PtrLong, ConstantInt::get(PtrLong->getType(),
                                                ~(0xFFULL << PointerTagShift)));
  }
  return UntaggedPtrLong;
}

Value *HWAddressSanitizer::getHwasanThreadSlotPtr(IRBuilder<> &IRB, Type *Ty) {
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  if (TargetTriple.isAArch64() && TargetTriple.isAndroid()) {
    // Android provides a fixed TLS slot for sanitizers. See TLS_SLOT_SANITIZER
    // in Bionic's libc/private/bionic_tls.h.
    Function *ThreadPointerFunc =
        Intrinsic::getDeclaration(M, Intrinsic::thread_pointer);
    Value *SlotPtr = IRB.CreatePointerCast(
        IRB.CreateConstGEP1_32(IRB.getInt8Ty(),
                               IRB.CreateCall(ThreadPointerFunc), 0x30),
        Ty->getPointerTo(0));
    return SlotPtr;
  }
  if (ThreadPtrGlobal)
    return ThreadPtrGlobal;

  return nullptr;
}

void HWAddressSanitizer::emitPrologue(IRBuilder<> &IRB, bool WithFrameRecord) {
  if (!Mapping.InTls)
    ShadowBase = getShadowNonTls(IRB);
  else if (!WithFrameRecord && TargetTriple.isAndroid())
    ShadowBase = getDynamicShadowIfunc(IRB);

  if (!WithFrameRecord && ShadowBase)
    return;

  Value *SlotPtr = getHwasanThreadSlotPtr(IRB, IntptrTy);
  assert(SlotPtr);

  Value *ThreadLong = IRB.CreateLoad(IntptrTy, SlotPtr);
  // Extract the address field from ThreadLong. Unnecessary on AArch64 with TBI.
  Value *ThreadLongMaybeUntagged =
      TargetTriple.isAArch64() ? ThreadLong : untagPointer(IRB, ThreadLong);

  if (WithFrameRecord) {
    Function *F = IRB.GetInsertBlock()->getParent();
    StackBaseTag = IRB.CreateAShr(ThreadLong, 3);

    // Prepare ring buffer data.
    Value *PC;
    if (TargetTriple.getArch() == Triple::aarch64)
      PC = readRegister(IRB, "pc");
    else
      PC = IRB.CreatePtrToInt(F, IntptrTy);
    Module *M = F->getParent();
    auto GetStackPointerFn = Intrinsic::getDeclaration(
        M, Intrinsic::frameaddress,
        IRB.getInt8PtrTy(M->getDataLayout().getAllocaAddrSpace()));
    Value *SP = IRB.CreatePtrToInt(
        IRB.CreateCall(GetStackPointerFn,
                       {Constant::getNullValue(IRB.getInt32Ty())}),
        IntptrTy);
    // Mix SP and PC.
    // Assumptions:
    // PC is 0x0000PPPPPPPPPPPP  (48 bits are meaningful, others are zero)
    // SP is 0xsssssssssssSSSS0  (4 lower bits are zero)
    // We only really need ~20 lower non-zero bits (SSSS), so we mix like this:
    //       0xSSSSPPPPPPPPPPPP
    SP = IRB.CreateShl(SP, 44);

    // Store data to ring buffer.
    Value *RecordPtr =
        IRB.CreateIntToPtr(ThreadLongMaybeUntagged, IntptrTy->getPointerTo(0));
    IRB.CreateStore(IRB.CreateOr(PC, SP), RecordPtr);

    // Update the ring buffer. Top byte of ThreadLong defines the size of the
    // buffer in pages, it must be a power of two, and the start of the buffer
    // must be aligned by twice that much. Therefore wrap around of the ring
    // buffer is simply Addr &= ~((ThreadLong >> 56) << 12).
    // The use of AShr instead of LShr is due to
    //   https://bugs.llvm.org/show_bug.cgi?id=39030
    // Runtime library makes sure not to use the highest bit.
    Value *WrapMask = IRB.CreateXor(
        IRB.CreateShl(IRB.CreateAShr(ThreadLong, 56), 12, "", true, true),
        ConstantInt::get(IntptrTy, (uint64_t)-1));
    Value *ThreadLongNew = IRB.CreateAnd(
        IRB.CreateAdd(ThreadLong, ConstantInt::get(IntptrTy, 8)), WrapMask);
    IRB.CreateStore(ThreadLongNew, SlotPtr);
  }

  if (!ShadowBase) {
    // Get shadow base address by aligning RecordPtr up.
    // Note: this is not correct if the pointer is already aligned.
    // Runtime library will make sure this never happens.
    ShadowBase = IRB.CreateAdd(
        IRB.CreateOr(
            ThreadLongMaybeUntagged,
            ConstantInt::get(IntptrTy, (1ULL << kShadowBaseAlignment) - 1)),
        ConstantInt::get(IntptrTy, 1), "hwasan.shadow");
    ShadowBase = IRB.CreateIntToPtr(ShadowBase, Int8PtrTy);
  }
}

Value *HWAddressSanitizer::readRegister(IRBuilder<> &IRB, StringRef Name) {
  Module *M = IRB.GetInsertBlock()->getParent()->getParent();
  Function *ReadRegister =
      Intrinsic::getDeclaration(M, Intrinsic::read_register, IntptrTy);
  MDNode *MD = MDNode::get(*C, {MDString::get(*C, Name)});
  Value *Args[] = {MetadataAsValue::get(*C, MD)};
  return IRB.CreateCall(ReadRegister, Args);
}

bool HWAddressSanitizer::instrumentLandingPads(
    SmallVectorImpl<Instruction *> &LandingPadVec) {
  for (auto *LP : LandingPadVec) {
    IRBuilder<> IRB(LP->getNextNode());
    IRB.CreateCall(
        HWAsanHandleVfork,
        {readRegister(IRB, (TargetTriple.getArch() == Triple::x86_64) ? "rsp"
                                                                      : "sp")});
  }
  return true;
}

static bool
maybeReachableFromEachOther(const SmallVectorImpl<IntrinsicInst *> &Insts,
                            const DominatorTree &DT) {
  // If we have too many lifetime ends, give up, as the algorithm below is N^2.
  if (Insts.size() > ClMaxLifetimes)
    return true;
  for (size_t I = 0; I < Insts.size(); ++I) {
    for (size_t J = 0; J < Insts.size(); ++J) {
      if (I == J)
        continue;
      if (isPotentiallyReachable(Insts[I], Insts[J], nullptr, &DT))
        return true;
    }
  }
  return false;
}

// static
bool HWAddressSanitizer::isStandardLifetime(const AllocaInfo &AllocaInfo,
                                            const DominatorTree &DT) {
  // An alloca that has exactly one start and end in every possible execution.
  // If it has multiple ends, they have to be unreachable from each other, so
  // at most one of them is actually used for each execution of the function.
  return AllocaInfo.LifetimeStart.size() == 1 &&
         (AllocaInfo.LifetimeEnd.size() == 1 ||
          (AllocaInfo.LifetimeEnd.size() > 0 &&
           !maybeReachableFromEachOther(AllocaInfo.LifetimeEnd, DT)));
}

bool HWAddressSanitizer::instrumentStack(
    bool ShouldDetectUseAfterScope,
    MapVector<AllocaInst *, AllocaInfo> &AllocasToInstrument,
    SmallVector<Instruction *, 4> &UnrecognizedLifetimes,
    DenseMap<AllocaInst *, std::vector<DbgVariableIntrinsic *>> &AllocaDbgMap,
    SmallVectorImpl<Instruction *> &RetVec, Value *StackTag,
    llvm::function_ref<const DominatorTree &()> GetDT,
    llvm::function_ref<const PostDominatorTree &()> GetPDT) {
  // Ideally, we want to calculate tagged stack base pointer, and rewrite all
  // alloca addresses using that. Unfortunately, offsets are not known yet
  // (unless we use ASan-style mega-alloca). Instead we keep the base tag in a
  // temp, shift-OR it into each alloca address and xor with the retag mask.
  // This generates one extra instruction per alloca use.

  unsigned int I = 0;

  for (auto &KV : AllocasToInstrument) {
    auto N = I++;
    auto *AI = KV.first;
    AllocaInfo &Info = KV.second;
    IRBuilder<> IRB(AI->getNextNode());

    // Replace uses of the alloca with tagged address.
    Value *Tag = getAllocaTag(IRB, StackTag, AI, N);
    Value *AILong = IRB.CreatePointerCast(AI, IntptrTy);

    Value *Replacement = tagPointer(IRB, AI->getType(), AILong, Tag);
    std::string Name =
        AI->hasName() ? AI->getName().str() : "alloca." + itostr(N);
    Replacement->setName(Name + ".hwasan");

    AI->replaceUsesWithIf(Replacement,
                          [AILong](Use &U) { return U.getUser() != AILong; });

    for (auto *DDI : AllocaDbgMap.lookup(AI)) {
      // Prepend "tag_offset, N" to the dwarf expression.
      // Tag offset logically applies to the alloca pointer, and it makes sense
      // to put it at the beginning of the expression.
      SmallVector<uint64_t, 8> NewOps = {dwarf::DW_OP_LLVM_tag_offset,
                                         retagMask(N)};
      for (size_t LocNo = 0; LocNo < DDI->getNumVariableLocationOps(); ++LocNo)
        if (DDI->getVariableLocationOp(LocNo) == AI)
          DDI->setExpression(DIExpression::appendOpsToArg(DDI->getExpression(),
                                                          NewOps, LocNo));
    }

    size_t Size = getAllocaSizeInBytes(*AI);
    auto TagEnd = [&](Instruction *Node) {
      IRB.SetInsertPoint(Node);
      Value *UARTag = getUARTag(IRB, StackTag);
      tagAlloca(IRB, AI, UARTag, Size);
    };
    bool StandardLifetime =
        UnrecognizedLifetimes.empty() && isStandardLifetime(Info, GetDT());
    if (ShouldDetectUseAfterScope && StandardLifetime) {
      IntrinsicInst *Start = Info.LifetimeStart[0];
      IRB.SetInsertPoint(Start->getNextNode());
      shadeAlloca(IRB, AI, Tag, Size);
      if (!forAllReachableExits(GetDT(), GetPDT(), Start, Info.LifetimeEnd,
                                RetVec, TagEnd)) {
        for (auto *End : Info.LifetimeEnd)
          End->eraseFromParent();
      }
    } else {
      shadeAlloca(IRB, AI, Tag, Size);
      for (auto *RI : RetVec)
        TagEnd(RI);
      if (!StandardLifetime) {
        for (auto &II : Info.LifetimeStart)
          II->eraseFromParent();
        for (auto &II : Info.LifetimeEnd)
          II->eraseFromParent();
      }
    }
  }
  for (auto &I : UnrecognizedLifetimes)
    I->eraseFromParent();
  return true;
}

bool HWAddressSanitizer::isInterestingAlloca(const AllocaInst &AI) {
  bool isInteresting =
      (AI.getAllocatedType()->isSized() &&
       // FIXME: instrument dynamic allocas, too
       AI.isStaticAlloca() &&
       // alloca() may be called with 0 size, ignore it.
       getAllocaSizeInBytes(AI) > 0 &&
       // We are only interested in allocas not promotable to registers.
       // Promotable allocas are common under -O0.
       !isAllocaPromotable(&AI) &&
       // inalloca allocas are not treated as static, and we don't want
       // dynamic alloca instrumentation for them as well.
       !AI.isUsedWithInAlloca() &&
       // swifterror allocas are register promoted by ISel
       !AI.isSwiftError()) &&
      // safe allocas are not interesting
      !(SSI && SSI->isSafe(AI));

  return isInteresting;
}

DenseMap<AllocaInst *, AllocaInst *> HWAddressSanitizer::padInterestingAllocas(
    const MapVector<AllocaInst *, AllocaInfo> &AllocasToInstrument) {
  DenseMap<AllocaInst *, AllocaInst *> AllocaToPaddedAllocaMap;
  for (auto &KV : AllocasToInstrument) {
    AllocaInst *AI = KV.first;
    uint64_t Size = getAllocaSizeInBytes(*AI);
    uint64_t AlignedSize = alignTo(Size, Mapping.getObjectAlignment());
    AI->setAlignment(
        Align(std::max(AI->getAlignment(), Mapping.getObjectAlignment())));
    if (Size != AlignedSize) {
      Type *AllocatedType = AI->getAllocatedType();
      if (AI->isArrayAllocation()) {
        uint64_t ArraySize =
            cast<ConstantInt>(AI->getArraySize())->getZExtValue();
        AllocatedType = ArrayType::get(AllocatedType, ArraySize);
      }
      Type *TypeWithPadding = StructType::get(
          AllocatedType, ArrayType::get(Int8Ty, AlignedSize - Size));
      auto *NewAI = new AllocaInst(
          TypeWithPadding, AI->getType()->getAddressSpace(), nullptr, "", AI);
      NewAI->takeName(AI);
      NewAI->setAlignment(AI->getAlign());
      NewAI->setUsedWithInAlloca(AI->isUsedWithInAlloca());
      NewAI->setSwiftError(AI->isSwiftError());
      NewAI->copyMetadata(*AI);
      auto *Bitcast = new BitCastInst(NewAI, AI->getType(), "", AI);
      AI->replaceAllUsesWith(Bitcast);
      AllocaToPaddedAllocaMap[AI] = NewAI;
    }
  }
  return AllocaToPaddedAllocaMap;
}

void HWAddressSanitizer::initializeFunctionDefinitionList(Module &M) {
  for (auto &F : M) {
    if (!F.isDeclaration())
      FunctionDefintionSet.insert(F.getGUID());
  }
}

bool HWAddressSanitizer::CallNeedsInstrumentation(const CallBase *CI) {
  if (!CI)
    return false;

  if (CI->isIndirectCall()) {
    return true;
  }

  if (!CI->getCalledFunction())
    return false;

  const Function *F = dyn_cast<Function>(CI->getCalledFunction());

  if (!F) {
    return false;
  }

  if (F->isIntrinsic()) {
    return false;
  }

  if (F->getName().str().rfind("llvm.experimental.vector.reduce.add", 0) == 0)
    return false; // skip this for now
  if (F->getName().startswith(ClMemoryAccessCallbackPrefix)) {
    return false; // one of our functions
  }

  bool foundFunction =
      std::find(FunctionDefintionSet.begin(), FunctionDefintionSet.end(),
                F->getGUID()) != FunctionDefintionSet.end();

  if (foundFunction) {
    return false;
  }
  return true;
}

bool HWAddressSanitizer::sanitizeFunction(
    Function &F, llvm::function_ref<const DominatorTree &()> GetDT,
    llvm::function_ref<const PostDominatorTree &()> GetPDT) {

  if (&F == HwasanCtorFunction)
    return false;

  if (!F.hasFnAttribute(Attribute::SanitizeHWAddress))
    return false;

  SmallVector<InterestingMemoryOperand, 16> OperandsToInstrument;
  SmallVector<MemIntrinsic *, 16> IntrinToInstrument;
  MapVector<AllocaInst *, AllocaInfo> AllocasToInstrument;
  SmallVector<Instruction *, 8> RetVec;
  SmallVector<Instruction *, 8> LandingPadVec;
  SmallVector<Instruction *, 4> UnrecognizedLifetimes;
  SmallVector<GetElementPtrInst *, 4> GEPsToInstrument;
  SmallVector<Instruction *, 4> CMPsToInstrument;
  SmallVector<CallInst *, 4> MallocsToInstruments;
  SmallVector<CallBase *, 4> CallsToInstruments;
  DenseMap<AllocaInst *, std::vector<DbgVariableIntrinsic *>> AllocaDbgMap;
  bool CallsReturnTwice = false;
  for (auto &BB : F) {
    for (auto &Inst : BB) {
      if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
        if (CI->canReturnTwice()) {
          CallsReturnTwice = true;
        }
      }
      if (InstrumentStack) {
        if (AllocaInst *AI = dyn_cast<AllocaInst>(&Inst)) {
          if (isInterestingAlloca(*AI))
            AllocasToInstrument.insert({AI, {}});
          continue;
        }
        auto *II = dyn_cast<IntrinsicInst>(&Inst);
        if (II && (II->getIntrinsicID() == Intrinsic::lifetime_start ||
                   II->getIntrinsicID() == Intrinsic::lifetime_end)) {
          AllocaInst *AI = findAllocaForValue(II->getArgOperand(1));
          if (!AI) {
            UnrecognizedLifetimes.push_back(&Inst);
            continue;
          }
          if (!isInterestingAlloca(*AI))
            continue;
          if (II->getIntrinsicID() == Intrinsic::lifetime_start)
            AllocasToInstrument[AI].LifetimeStart.push_back(II);
          else
            AllocasToInstrument[AI].LifetimeEnd.push_back(II);
          continue;
        }
      }

      if (GetElementPtrInst *GEPI = dyn_cast<GetElementPtrInst>(&Inst)) {
        GEPsToInstrument.push_back(GEPI);
      }

      if (Instruction::ICmp == Inst.getOpcode()) {
        CMPsToInstrument.push_back(&Inst);
      }

      if (CallInst *CI = dyn_cast<CallInst>(&Inst)) {
        if (isAllocationFunction(CI) || isAllocationWrapper(CI))
          MallocsToInstruments.push_back(CI);
      }

      if (isa<ReturnInst>(Inst)) {
        if (CallInst *CI = Inst.getParent()->getTerminatingMustTailCall())
          RetVec.push_back(CI);
        else
          RetVec.push_back(&Inst);
      } else if (isa<ResumeInst, CleanupReturnInst>(Inst)) {
        RetVec.push_back(&Inst);
      }

      if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&Inst)) {
        for (Value *V : DVI->location_ops()) {
          if (auto *Alloca = dyn_cast_or_null<AllocaInst>(V))
            if (!AllocaDbgMap.count(Alloca) ||
                AllocaDbgMap[Alloca].back() != DVI)
              AllocaDbgMap[Alloca].push_back(DVI);
        }
      }

      if (InstrumentLandingPads && isa<LandingPadInst>(Inst))
        LandingPadVec.push_back(&Inst);
#ifndef DISABLE_CHECKS
      getInterestingMemoryOperands(&Inst, OperandsToInstrument);

      if (MemIntrinsic *MI = dyn_cast<MemIntrinsic>(&Inst))
        if (!ignoreMemIntrinsic(MI))
          IntrinToInstrument.push_back(MI);
#endif
    }
  }

  for (auto &Operand : OperandsToInstrument)
    Operand.shaded = !isNotPartOfStruct(Operand.getPtr());

  initializeCallbacks(*F.getParent());

  bool Changed = false;

  if (!LandingPadVec.empty())
    Changed |= instrumentLandingPads(LandingPadVec);

  if (AllocasToInstrument.empty() && F.hasPersonalityFn() &&
      F.getPersonalityFn()->getName() == kHwasanPersonalityThunkName) {
    // __hwasan_personality_thunk is a no-op for functions without an
    // instrumented stack, so we can drop it.
    F.setPersonalityFn(nullptr);
    Changed = true;
  }

  if (AllocasToInstrument.empty() && OperandsToInstrument.empty() &&
      IntrinToInstrument.empty())
    return Changed;

  assert(!ShadowBase);

  Instruction *InsertPt = &*F.getEntryBlock().begin();
  IRBuilder<> EntryIRB(InsertPt);
  emitPrologue(EntryIRB,
               /*WithFrameRecord*/ ClRecordStackHistory &&
                   Mapping.WithFrameRecord && !AllocasToInstrument.empty());

  if (!AllocasToInstrument.empty()) {
    Value *StackTag =
        ClGenerateTagsWithCalls ? nullptr : getStackBaseTag(EntryIRB);
    // Calls to functions that may return twice (e.g. setjmp) confuse the
    // postdominator analysis, and will leave us to keep memory tagged after
    // function return. Work around this by always untagging at every return
    // statement if return_twice functions are called.
    instrumentStack(DetectUseAfterScope && !CallsReturnTwice,
                    AllocasToInstrument, UnrecognizedLifetimes, AllocaDbgMap,
                    RetVec, StackTag, GetDT, GetPDT);
  }
  // Pad and align each of the allocas that we instrumented to stop small
  // uninteresting allocas from hiding in instrumented alloca's padding and so
  // that we have enough space to store real tags for short granules.
  DenseMap<AllocaInst *, AllocaInst *> AllocaToPaddedAllocaMap =
      padInterestingAllocas(AllocasToInstrument);

  if (!AllocaToPaddedAllocaMap.empty()) {
    for (auto &BB : F) {
      for (auto &Inst : BB) {
        if (auto *DVI = dyn_cast<DbgVariableIntrinsic>(&Inst)) {
          SmallDenseSet<Value *> LocationOps(DVI->location_ops().begin(),
                                             DVI->location_ops().end());
          for (Value *V : LocationOps) {
            if (auto *AI = dyn_cast_or_null<AllocaInst>(V)) {
              if (auto *NewAI = AllocaToPaddedAllocaMap.lookup(AI))
                DVI->replaceVariableLocationOp(V, NewAI);
            }
          }
        }
      }
    }
    for (auto &P : AllocaToPaddedAllocaMap)
      P.first->eraseFromParent();
  }

  // If we split the entry block, move any allocas that were originally in the
  // entry block back into the entry block so that they aren't treated as
  // dynamic allocas.
  if (EntryIRB.GetInsertBlock() != &F.getEntryBlock()) {
    InsertPt = &*F.getEntryBlock().begin();
    for (Instruction &I :
         llvm::make_early_inc_range(*EntryIRB.GetInsertBlock())) {
      if (auto *AI = dyn_cast<AllocaInst>(&I))
        if (isa<ConstantInt>(AI->getArraySize()))
          I.moveBefore(InsertPt);
    }
  }

#ifndef DISABLE_GEP_INSTRUMENTATION
  for (auto &ConstGEPI : ConstGEPsToInstrument) {
    instrumentConstGEP(ConstGEPI);
  }

  // if (instrumentGEPs) {
  for (auto &GEPI : GEPsToInstrument) {
    instrumentGEP(GEPI);
  }
  // }
#endif

  for (auto &CMP : CMPsToInstrument) {
    instrumentCMP(CMP);
  }

  for (auto &Operand : OperandsToInstrument)
    instrumentMemAccess(Operand);

  if (ClInstrumentMemIntrinsics && !IntrinToInstrument.empty()) {
    for (auto Inst : IntrinToInstrument)
      instrumentMemIntrinsic(cast<MemIntrinsic>(Inst));
  }

  for (auto &CI : MallocsToInstruments) {
    instrumentMalloc(CI);
  }

  for (auto &CI : CallsToInstruments) {
    for (unsigned i = 0; i < CI->getNumOperands(); i++) {
      Value *pointeroperand = CI->getOperand(i);
      Type *poType = pointeroperand->getType();
      if (poType->isPointerTy()) {
        IRBuilder<> IRB(CI);
        IRB.CreateCall(HwasanTestFreedFunc,
                       {IRB.CreatePointerCast(pointeroperand, Int8PtrTy)});
      }
    }
  }

  ShadowBase = nullptr;
  StackBaseTag = nullptr;

  return true;
}

void HWAddressSanitizer::instrumentGlobal(GlobalVariable *GV, uint8_t Tag) {
  LLVM_DEBUG(dbgs() << "instrumentGlobal " << *GV << " tag " << (int)Tag
                    << "\n");
  assert(!UsePageAliases);
  Constant *Initializer = GV->getInitializer();
  uint64_t SizeInBytes =
      M.getDataLayout().getTypeAllocSize(Initializer->getType());
  uint64_t NewSize = alignTo(SizeInBytes, Mapping.getObjectAlignment());
  if (SizeInBytes != NewSize) {
    // Pad the initializer out to the next multiple of 16 bytes and add the
    // required short granule tag.
    std::vector<uint8_t> Init(NewSize - SizeInBytes, 0);
    Init.back() = Tag;
    Constant *Padding = ConstantDataArray::get(*C, Init);
    Initializer = ConstantStruct::getAnon({Initializer, Padding});
  }

  auto *NewGV = new GlobalVariable(M, Initializer->getType(), GV->isConstant(),
                                   GlobalValue::ExternalLinkage, Initializer,
                                   GV->getName() + ".hwasan");

  if (dyn_cast<StructType>(GV->getValueType())) {
    StructGlobals.push_back(NewGV);

    for (auto User : GV->users()) {
      if (auto CExpr = dyn_cast<ConstantExpr>(User)) {
        if (CExpr->getOpcode() == llvm::Instruction::GetElementPtr) {
          ConstGEPsToInstrument.push_back(CExpr);
        }
      }
    }
  }

  NewGV->copyAttributesFrom(GV);
  NewGV->setLinkage(GlobalValue::PrivateLinkage);
  NewGV->copyMetadata(GV, 0);
  NewGV->setAlignment(
      MaybeAlign(std::max(GV->getAlignment(), Mapping.getObjectAlignment())));

  // It is invalid to ICF two globals that have different tags. In the case
  // where the size of the global is a multiple of the tag granularity the
  // contents of the globals may be the same but the tags (i.e. symbol values)
  // may be different, and the symbols are not considered during ICF. In the
  // case where the size is not a multiple of the granularity, the short granule
  // tags would discriminate two globals with different tags, but there would
  // otherwise be nothing stopping such a global from being incorrectly ICF'd
  // with an uninstrumented (i.e. tag 0) global that happened to have the short
  // granule tag in the last byte.
  NewGV->setUnnamedAddr(GlobalValue::UnnamedAddr::None);

  // Descriptor format (assuming little-endian):
  // bytes 0-3: relative address of global
  // bytes 4-6: size of global (16MB ought to be enough for anyone, but in case
  // it isn't, we create multiple descriptors)
  // byte 7: tag
  auto *DescriptorTy = StructType::get(Int32Ty, Int32Ty);
  const uint64_t MaxDescriptorSize = 0xfffff0;
  for (uint64_t DescriptorPos = 0; DescriptorPos < SizeInBytes;
       DescriptorPos += MaxDescriptorSize) {
    auto *Descriptor =
        new GlobalVariable(M, DescriptorTy, true, GlobalValue::PrivateLinkage,
                           nullptr, GV->getName() + ".hwasan.descriptor");
    auto *GVRelPtr = ConstantExpr::getTrunc(
        ConstantExpr::getAdd(
            ConstantExpr::getSub(
                ConstantExpr::getPtrToInt(NewGV, Int64Ty),
                ConstantExpr::getPtrToInt(Descriptor, Int64Ty)),
            ConstantInt::get(Int64Ty, DescriptorPos)),
        Int32Ty);
    uint32_t Size = std::min(SizeInBytes - DescriptorPos, MaxDescriptorSize);
    auto *SizeAndTag = ConstantInt::get(Int32Ty, Size | (uint32_t(Tag) << 24));
    Descriptor->setComdat(NewGV->getComdat());
    Descriptor->setInitializer(ConstantStruct::getAnon({GVRelPtr, SizeAndTag}));
    Descriptor->setSection("hwasan_globals");
    Descriptor->setMetadata(LLVMContext::MD_associated,
                            MDNode::get(*C, ValueAsMetadata::get(NewGV)));
    appendToCompilerUsed(M, Descriptor);
  }

  Constant *Aliasee = ConstantExpr::getIntToPtr(
      ConstantExpr::getAdd(
          ConstantExpr::getPtrToInt(NewGV, Int64Ty),
          ConstantInt::get(Int64Ty, uint64_t(Tag) << PointerTagShift)),
      GV->getType());
  auto *Alias = GlobalAlias::create(GV->getValueType(), GV->getAddressSpace(),
                                    GV->getLinkage(), "", Aliasee, &M);
  Alias->setVisibility(GV->getVisibility());
  Alias->takeName(GV);
  GV->replaceAllUsesWith(Alias);
  GV->eraseFromParent();
}

static DenseSet<GlobalVariable *> getExcludedGlobals(Module &M) {
  NamedMDNode *Globals = M.getNamedMetadata("llvm.asan.globals");
  if (!Globals)
    return DenseSet<GlobalVariable *>();
  DenseSet<GlobalVariable *> Excluded(Globals->getNumOperands());
  for (auto MDN : Globals->operands()) {
    // Metadata node contains the global and the fields of "Entry".
    assert(MDN->getNumOperands() == 5);
    auto *V = mdconst::extract_or_null<Constant>(MDN->getOperand(0));
    // The optimizer may optimize away a global entirely.
    if (!V)
      continue;
    auto *StrippedV = V->stripPointerCasts();
    auto *GV = dyn_cast<GlobalVariable>(StrippedV);
    if (!GV)
      continue;
    ConstantInt *IsExcluded = mdconst::extract<ConstantInt>(MDN->getOperand(4));
    if (IsExcluded->isOne())
      Excluded.insert(GV);
  }
  return Excluded;
}

void HWAddressSanitizer::instrumentGlobals() {
  auto ExcludedGlobals = getExcludedGlobals(M);
  for (GlobalVariable &GV : M.globals()) {
    if (ExcludedGlobals.count(&GV))
      continue;

    if (GV.isDeclarationForLinker() || GV.getName().startswith("llvm.") ||
        GV.isThreadLocal())
      continue;

    // Common symbols can't have aliases point to them, so they can't be tagged.
    if (GV.hasCommonLinkage())
      continue;

    // Globals with custom sections may be used in __start_/__stop_ enumeration,
    // which would be broken both by adding tags and potentially by the extra
    // padding/alignment that we insert.
    if (GV.hasSection())
      continue;

    Globals.push_back(&GV);
  }

  MD5 Hasher;
  Hasher.update(M.getSourceFileName());
  MD5::MD5Result Hash;
  Hasher.final(Hash);
  uint8_t Tag = Hash[0];
  for (GlobalVariable *GV : Globals) {
    Tag &= TagMaskByte;
    // Skip tag 0 in order to avoid collisions with untagged memory.
    if (Tag == 0)
      Tag = 1;
    Tag += 1 << 4ULL;
    Tag = Tag & 0xF0;
    instrumentGlobal(GV, Tag);
  }
}

void HWAddressSanitizer::applyShading(IRBuilder<> &IRB, StructType *type,
                                      Value *AI) {
#ifdef DISABLE_SHADING_ALLOCATIONS
  return;
#endif

  if (containsUnion(type)) {
    return;
  }

  std::vector<std::tuple<int, int>> shadeVector =
      getMetadataLayoutFromType(type);

  Value *CastedBaseAddress = IRB.CreatePointerCast(AI, Int8PtrTy);
  unsigned int offset = 0;
  for (unsigned i = 0; i < shadeVector.size(); ++i) {
    Value *TargetAddress = IRB.CreateInBoundsGEP(
        Int8Ty, CastedBaseAddress, ConstantInt::get(Int32Ty, offset));
    Value *JustShade = IRB.CreateTrunc(
        ConstantInt::get(Int8Ty, std::get<0>(shadeVector[i])), IRB.getInt8Ty());
    IRB.CreateCall(HwasanShadeMemoryFunc,
                   {TargetAddress, JustShade,
                    ConstantInt::get(IntptrTy, std::get<1>(shadeVector[i]))});
    offset += std::get<1>(shadeVector[i]);
  }
}

// Takes a type as input and calls recursively until subtypes are resolved
// Return the deepest nesting
unsigned long HWAddressSanitizerPass::getLongestPath(Type *type) {
  LLVM_DEBUG(dbgs() << "getLongestPath for " << *type << "\n");
  static std::map<Type *, unsigned long> memo;

  if (!type->isStructTy()) {
    LLVM_DEBUG(dbgs() << "!isStructTy, return 1\n");
    return 1;
  }

  if (memo.count(type) > 0) {
    LLVM_DEBUG(dbgs() << "Known: return " << memo[type] << "\n");
    return memo[type];
  }

  auto nContainedTypes = type->getNumContainedTypes();
  unsigned long max_sub_length = 0;
  for (unsigned int i = 0; i < nContainedTypes; i++) {
    unsigned long sub_length = getLongestPath(type->getContainedType(i));
    if (sub_length > max_sub_length) {
      max_sub_length = sub_length;
    }
  }
  memo[type] = max_sub_length + 1;
  return memo[type];
}

// Takes a type as input and calls recursively until subtypes are resolved
// Return the total number of members
unsigned long HWAddressSanitizerPass::getSubTypes(Type *type) {
  LLVM_DEBUG(dbgs() << "getSubTypes for " << *type << "\n");
  static std::map<Type *, unsigned long> memo;
  if (!type->isStructTy()) {
    LLVM_DEBUG(dbgs() << "!isStructTy, return 1\n");
    return 1;
  }

  if (memo.count(type) > 0) {
    LLVM_DEBUG(dbgs() << "Known: return " << memo[type] << "\n");
    return memo[type];
  }

  auto nContainedTypes = type->getNumContainedTypes();
  memo[type] = 0;
  for (unsigned int i = 0; i < nContainedTypes; i++) {
    memo[type] += getSubTypes(type->getContainedType(i));
  }
  LLVM_DEBUG(dbgs() << "Return total subtypes: " << memo[type] << "\n");
  return memo[type];
}

void HWAddressSanitizer::instrumentPersonalityFunctions() {
  // We need to untag stack frames as we unwind past them. That is the job of
  // the personality function wrapper, which either wraps an existing
  // personality function or acts as a personality function on its own. Each
  // function that has a personality function or that can be unwound past has
  // its personality function changed to a thunk that calls the personality
  // function wrapper in the runtime.
  MapVector<Constant *, std::vector<Function *>> PersonalityFns;
  for (Function &F : M) {
    if (F.isDeclaration() || !F.hasFnAttribute(Attribute::SanitizeHWAddress))
      continue;

    if (F.hasPersonalityFn()) {
      PersonalityFns[F.getPersonalityFn()->stripPointerCasts()].push_back(&F);
    } else if (!F.hasFnAttribute(Attribute::NoUnwind)) {
      PersonalityFns[nullptr].push_back(&F);
    }
  }

  if (PersonalityFns.empty())
    return;

  FunctionCallee HwasanPersonalityWrapper = M.getOrInsertFunction(
      "__hwasan_personality_wrapper", Int32Ty, Int32Ty, Int32Ty, Int64Ty,
      Int8PtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy, Int8PtrTy);
  FunctionCallee UnwindGetGR = M.getOrInsertFunction("_Unwind_GetGR", VoidTy);
  FunctionCallee UnwindGetCFA = M.getOrInsertFunction("_Unwind_GetCFA", VoidTy);

  for (auto &P : PersonalityFns) {
    std::string ThunkName = kHwasanPersonalityThunkName;
    if (P.first)
      ThunkName += ("." + P.first->getName()).str();
    FunctionType *ThunkFnTy = FunctionType::get(
        Int32Ty, {Int32Ty, Int32Ty, Int64Ty, Int8PtrTy, Int8PtrTy}, false);
    bool IsLocal = P.first && (!isa<GlobalValue>(P.first) ||
                               cast<GlobalValue>(P.first)->hasLocalLinkage());
    auto *ThunkFn = Function::Create(ThunkFnTy,
                                     IsLocal ? GlobalValue::InternalLinkage
                                             : GlobalValue::LinkOnceODRLinkage,
                                     ThunkName, &M);
    if (!IsLocal) {
      ThunkFn->setVisibility(GlobalValue::HiddenVisibility);
      ThunkFn->setComdat(M.getOrInsertComdat(ThunkName));
    }

    auto *BB = BasicBlock::Create(*C, "entry", ThunkFn);
    IRBuilder<> IRB(BB);
    CallInst *WrapperCall = IRB.CreateCall(
        HwasanPersonalityWrapper,
        {ThunkFn->getArg(0), ThunkFn->getArg(1), ThunkFn->getArg(2),
         ThunkFn->getArg(3), ThunkFn->getArg(4),
         P.first ? IRB.CreateBitCast(P.first, Int8PtrTy)
                 : Constant::getNullValue(Int8PtrTy),
         IRB.CreateBitCast(UnwindGetGR.getCallee(), Int8PtrTy),
         IRB.CreateBitCast(UnwindGetCFA.getCallee(), Int8PtrTy)});
    WrapperCall->setTailCall();
    IRB.CreateRet(WrapperCall);

    for (Function *F : P.second)
      F->setPersonalityFn(ThunkFn);
  }
}

void HWAddressSanitizer::ShadowMapping::init(Triple &TargetTriple,
                                             bool InstrumentWithCalls) {
  Scale = kDefaultShadowScale;
  if (TargetTriple.isOSFuchsia()) {
    // Fuchsia is always PIE, which means that the beginning of the address
    // space is always available.
    InGlobal = false;
    InTls = false;
    Offset = 0;
    WithFrameRecord = true;
  } else if (ClMappingOffset.getNumOccurrences() > 0) {
    InGlobal = false;
    InTls = false;
    Offset = ClMappingOffset;
    WithFrameRecord = false;
  } else if (ClEnableKhwasan) {
    InGlobal = false;
    InTls = false;
    Offset = 0;
    WithFrameRecord = false;
  } else if (ClWithIfunc) {
    InGlobal = true;
    InTls = false;
    Offset = kDynamicShadowSentinel;
    WithFrameRecord = false;
  } else if (0) {
    InGlobal = false;
    InTls = true;
    Offset = kDynamicShadowSentinel;
    WithFrameRecord = true;
  } else {
    InGlobal = false;
    InTls = false;
    Offset = 0;
    WithFrameRecord = false;
  }
}
