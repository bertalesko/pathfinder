What Grand Expeditions are

A Grand Expedition is the scaled-up Expedition: an entire region (not a corner of a map) that functions as one giant dig site. It's the core reward engine of the reworked Expedition system, tied to the Runes of Aldur league, and reached through an ocean area southeast of the Ruins of Kingsmarch. You unlock it via the questline (first Tier 1 map → Farrow → Kingsmarch → logbooks reveal ocean islands → through four deadly bosses → The Aberration), then farm it indefinitely. It scales with Waystone tier, so Tier 15–16 maps get reserved for it, and the top end pays 20–30 Divines a run.

The loop itself: you place explosives to blow up buried remnants, which reanimate into runic monster packs you kill for loot. The hard constraint is that each charge chains off the previous one — you're drawing one connected path, not scattering bombs — and only detonated remnants pay out. A Grand Expedition gives 15+ charges.

The proliferation mechanic (the heart of ordering)

This is what makes order matter, and it's the rule your script models:

Certain slots are crowned (gold-framed). The rune in a crowned slot proliferates — it carries onto every subsequent explosion in the chain, so all later monster packs also carry that rune's effect.
Effects accumulate down the chain. Fire your good proliferators early and they buff everything after them, so the end of the chain collects the full stacked bonus. This is why the instinct is "build toward a rich finale."
Same runes don't stack. A duplicate proliferating rune contributes nothing — only the first of its type counts. So variety of crowned runes matters, not quantity.
Only the crowned rune transfers, not the whole remnant's contents. A great rune in a non-crowned slot stays local.
Some runes are traps. Oath (and Wisdom, Bait) have low or negative loot value and should never be proliferated — in-game Oath actively poisons the chain with loot-less waves. Your Avoid flag encodes exactly this.

Rune proliferation tiers, for reference: Opulent (SS) > Power/Bond/Death (S) > Time/Rebirth (A), with Oath an S-value trap you avoid, and Wisdom/Bait neutral-to-bad.

How to order remnants efficiently

The naive "biggest stone last" heuristic turns out to be wrong once you model everything together — which is the real finding from building the optimizer. The correct objective is: maximize the chain's stacked proliferation reward, subject to the charge budget and travel distance. Concretely:

Front-load distinct, high-value proliferators. Get Opulent/Power/Bond/Death/Time into the stack as early as the layout allows, so they multiply the most downstream remnants. Different runes — duplicates are wasted.
Don't force the biggest stone last. A high-slot stone is often a better proliferator spent early than a collector spent late, especially if it's far away. Let the endpoint be whichever stone actually scores best as the collector — sometimes a smaller, well-placed one.
Route is value-shaped, not distance-shaped. The path should wind through crowned proliferators on the way to the finale, but every detour costs charges — so the optimizer trades reward against budget rather than taking the shortest path.
Never proliferate Avoid runes. Visit those stones only for their local loot if at all; keep Oath/Wisdom/Bait out of the stack.
Respect the budget. A greedy crown-collecting route that runs dry before the finale loses the whole stacked payoff. Feasibility first, then maximize.