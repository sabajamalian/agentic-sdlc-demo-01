# Forecasting working session

**Date:** 2026-07-15
**Duration:** 48 minutes
**Attendees:** Priya (Supply Planning), Marcus (Data Science), Dana (Retail Ops), Sam (Engineering)
**Recording:** auto-transcribed, lightly cleaned up

---

**Priya:** Let's start. Sam, you're joining late next week for the platform stuff?

**Sam:** Wednesday. I'm out Monday and Tuesday.

**Priya:** Fine, we'll move the platform sync. Let's get into the forecast.

**Marcus:** So where we are today: we've got the SARIMAX model running against the
aggregate daily series, backtested over five rolling origins with a fourteen day
horizon. It's landing around nine point three percent MAPE. The seasonal naive
baseline is at ten point three. So we're beating the baseline, but not by a lot.

**Dana:** Nine percent on what exactly? Because when I look at the store level,
some of these are way off.

**Marcus:** On the total. Everything's aggregated right now.

**Dana:** That's the thing I keep running into. The aggregate number looks fine
and then I go to actually order for a specific SKU and it's nowhere near. Last
month the two-liter went short three days in a row while the aggregate error was
under five percent.

**Priya:** Because the errors cancel out.

**Dana:** Right. One SKU over, one SKU under, the total looks great, and my
replenishment is still wrong.

**Marcus:** Yeah, that's fair. Aggregate MAPE is genuinely hiding the variance.

**Dana:** What I need is per-SKU. I want to open the report and see, for each SKU,
what the error is, and I want to know which ones are the worst so I know where to
apply judgement. Right now I'm flying blind on anything below the total.

**Priya:** Can we do that?

**Marcus:** We can. The backtest already refits per fold, so running it per SKU is
mostly a loop and then a table. The thing I'd want is the per-SKU numbers *and*
the aggregate side by side, so we can see whether a model that's better on the
total is actually worse on individual items.

**Dana:** Sorted worst first, please. And I want it in the report artifact, not
something I have to run a notebook for.

**Marcus:** Noted.

**Priya:** OK. Second thing. Marcus, you mentioned Prophet a few weeks ago.

**Marcus:** I did. So SARIMAX is doing fine on the weekly seasonality but it's
struggling around the yearly pattern, especially the run up to the holidays. It
tends to under-forecast the ramp and then over-forecast the drop afterwards.
Prophet handles multiple seasonalities and changepoints more gracefully, and it's
much more forgiving about the trend shifting.

**Sam:** Is that a heavy dependency? I don't want CI install times to blow up.

**Marcus:** It pulls in a compiled backend, so it's not free. But we've already
set the project up so extra model dependencies go in an optional group. It
wouldn't be in the default install.

**Sam:** As long as it's optional and CI still works without it, I'm fine.

**Marcus:** It'd go through the model registry like everything else, so it's a
new class plus one registry entry. Then we backtest it against SARIMAX and the
naive baseline on identical splits and see if it's actually better. If it's not,
we don't ship it. I don't want to add a dependency on vibes.

**Priya:** Agreed, it has to earn its place. What would convince you?

**Marcus:** Beating SARIMAX on the same rolling origins. If it wins, it becomes
the champion. If it loses, we've still got the comparison written down and we
stop having this conversation every quarter.

**Priya:** Good. Third thing, and this is the one that actually bothers me.
Dana, tell them about the promo weekend.

**Dana:** So we ran the summer promotion the weekend of June the twentieth.
Volume roughly doubled. The forecast had no idea. It just treated it as a normal
Saturday and we were short everywhere.

**Marcus:** Right, because the model doesn't see promotions at all. We have the
`on_promotion` flag in the raw data. It's in the CSV. Nothing uses it.

**Priya:** Why not?

**Marcus:** Honestly, it just wasn't wired up. Same for holidays. We have no
holiday calendar in the feature set, so Thanksgiving and the Fourth of July look
like ordinary days with weird numbers.

**Dana:** And those are the days I care about most. Nobody gets fired over a
Tuesday in March.

**Marcus:** So the work is a calendar feature set. Holiday flags, some notion of
days until the next holiday and days since the last one, and promotion features
off the existing flag.

**Sam:** Careful with that. If you build a "days until next promotion" feature
you're assuming you know the future promo schedule.

**Marcus:** For promotions we actually do, they're planned six weeks out, so
that's legitimately known at forecast time. Same for holidays, obviously. But
you're right that anything derived from actual sales during a promotion is not
known, and that's the trap. If I build a "typical lift during promotions"
feature off historical units I have to make sure it's computed on the training
window only.

**Sam:** That's exactly what I mean. That kind of thing looks amazing in backtest
and then does nothing in production.

**Marcus:** Agreed. We already have look-ahead tests for the lag features. I'd
want the same for anything new here: perturb the last target value, assert
nothing earlier moves.

**Priya:** Which holidays? We're US only right now but Canada is on the roadmap.

**Marcus:** US federal to start. I'd rather not build a general calendar
abstraction for a market we don't serve yet.

**Priya:** Fine, US only. We'll revisit when Canada is real.

**Dana:** What about weather? Ice cream sales are obviously weather driven.

**Marcus:** That's a much bigger piece of work. We'd need a weather data feed,
and forecast weather rather than historical weather, otherwise it's leakage
again. I'd park it.

**Priya:** Park it. Not this quarter.

**Sam:** One thing I want to raise separately, not for this list. We should
probably move the training runs off the shared runner at some point, the eval
job is getting slow.

**Priya:** Separate conversation, take it to the platform sync.

**Priya:** Last thing, and I don't think we resolved it. Do we retrain on a
schedule or only when the eval gate flags a regression?

**Marcus:** I lean scheduled, weekly. But there's an argument for event-driven.

**Dana:** Doesn't retraining weekly mean the forecast changes under me even when
nothing's wrong?

**Marcus:** It might, yeah.

**Priya:** Let's leave that open. I want to see the per-SKU numbers before we
decide anything about retraining cadence, because that'll tell us how stable
these models actually are.

**Marcus:** Fair.

**Priya:** So: per-SKU reporting, Prophet as a candidate model, and the
holiday and promotion features. Weather is parked, retraining cadence is
unresolved, and Sam's runner thing goes to platform.

**Marcus:** That's my list too.

**Priya:** Good. Same time in two weeks.

---

*Transcript ends.*
