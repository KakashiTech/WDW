#!/usr/bin/env julia
using WDW

const Know = WDW.Knowledge
const Cat = WDW.Category
const Sh = WDW.Sheaves

# Deterministic topology (Alexandrov chain) for clean Heyting behavior
X = [1,2,3,4]
opens = [Int[], [4], [3,4], [2,3,4], [1,2,3,4]]
ts = Know.TopSpace(X, opens)

# Heyting elements
hA = Know.HeytingOpen(ts, [3,4])
hB = Know.HeytingOpen(ts, [2,3,4])
hTop = Know.heyting_top(ts)
hBot = Know.heyting_bot(ts)

# Evaluate basic laws
law_and_unit = Set(Know.heyting_and(hA, hTop).U) == Set(hA.U)
law_imp_refl = Set(Know.heyting_imply(hA, hA).U) == Set(hTop.U)
law_bot_leq = Know.heyting_leq(hBot, hA)

# Subobject classifier artifacts
Xset = Cat.FinSet(X)
U = [1,3]
chi = Cat.characteristic(Xset, U)
pb_dom, π1, π2 = Cat.pullback(chi, Cat.true_map())
sub_size = length(pb_dom.elements)

# Sheaf partial knowledge integration
opens2 = [Int[], [1], [2], [1,2], [2,3], [1,2,3]]
ts2 = Know.TopSpace(X, opens2)
sheaf = Sh.ConstantSheaf(ts2, [0,1])
cover = [[1,2], [2,3]]
sections_ok = [1,1]
sections_bad = [0,1]

ok1, p1 = Sh.glue_via_partials(sheaf, cover, sections_ok)
ok2, p2 = Sh.glue_via_partials(sheaf, cover, sections_bad)

isdir("bench") || mkpath("bench")
open("bench/phase1_heyting_truths.csv", "w") do io
    println(io, "property,value")
    println(io, "and_unit,$law_and_unit")
    println(io, "imp_reflexive,$law_imp_refl")
    println(io, "bot_leq,$law_bot_leq")
end

open("bench/phase1_sets_omega.csv", "w") do io
    println(io, "metric,value")
    println(io, "sub_size,$sub_size")
    println(io, "char_true_at_1,$(chi.map[1])")
    println(io, "char_true_at_2,$(chi.map[2])")
    println(io, "char_true_at_3,$(chi.map[3])")
    println(io, "char_true_at_4,$(chi.map[4])")
end

open("bench/phase1_sheaves_partials.csv", "w") do io
    println(io, "case,ok,domain_size,value_or_nan")
    println(io, "consistent,$ok1,$(ok1 ? length(p1.domain) : 0),$(ok1 ? p1.value : NaN)")
    println(io, "inconsistent,$ok2,$(ok2 ? length(p2.domain) : 0),$(ok2 ? p2.value : NaN)")
end

open("bench/phase1_heyting_certificate.txt", "w") do io
    println(io, "WDW++ Phase 1′ Certificate: Heyting, Ω, Partial Knowledge")
    println(io, "heyting_and_unit=$(law_and_unit)")
    println(io, "heyting_imp_reflexive=$(law_imp_refl)")
    println(io, "heyting_bot_leq=$(law_bot_leq)")
    println(io, "subobject_pullback_size=$(sub_size)")
    println(io, "sheaf_glue_consistent=$(ok1)")
    println(io, "sheaf_glue_inconsistent=$(ok2)")
end
