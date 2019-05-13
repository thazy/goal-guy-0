// Copyright (c) 2019, The Emergent Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// goal-guy-0.go is a simple Phase 0 begin-with-success model to serve as a basis for learning motor and then instrumental actions
// based on the key idea of striving toward desired arbitrary outcome states. Phase 0 uses all localist reps
// and standard error driven learning.  Phase 0.5 will convert some reps to full distributed.
// Phase 1 will move to DeepLeabra implementation.
package main

import (
	"fmt"
	"log"
	"math/rand"
	"time"

	"github.com/chewxy/math32"

	"github.com/emer/emergent/emer"
	"github.com/emer/emergent/erand"
	"github.com/emer/emergent/netview"
	"github.com/emer/emergent/patgen"
	"github.com/emer/emergent/prjn"
	"github.com/emer/emergent/relpos"
	"github.com/emer/emergent/timer"

	"github.com/emer/etable/eplot"
	"github.com/emer/etable/etable"
	"github.com/emer/etable/etensor"

	"github.com/emer/leabra/leabra"

	"github.com/goki/gi/gi"
	"github.com/goki/gi/gimain"
	"github.com/goki/gi/giv"
	"github.com/goki/gi/svg"
	"github.com/goki/gi/units"
	"github.com/goki/ki/ki"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// todo:
//
// * make etable/eplot2.Plot which encapsulates the SVGEditor and
// shows its columns -- basically replicating behavior of C++
// fGraphView for easier dynamic selection of what to plot, how to
// plot it, etc.
// Trick is how to make it slos customizable via code..
//
// * LogTrainTrial, LogTestTrial, LogTestCycle and assoc. plots
//
// * etable/eview.TableView (gridview -- is there a gonum version?)
// and TableEdit (spreadsheet-like editor)
// and show these in another tab for the input patterns
//

// this is the stub main for gogi that calls our actual mainrun function, at end of file
func main() {
	gimain.Main(func() {
		mainrun()
	})
}

// DefaultParams are the initial default parameters for this simulation
var DefaultParams = emer.ParamStyle{
	{"Prjn", emer.Params{
		"Prjn.Learn.Norm.On":     1,
		"Prjn.Learn.Momentum.On": 1,
		"Prjn.Learn.WtBal.On":    0,
	}},
	// TODO: below appears to be old, bad syntax
	// "Layer": {
	// 	//"Layer.Inhib.Layer.Gi": 1.8, // this is the default
	// 	"Layer.Inhib.Layer.Gi": 2.8, // going for k = 1 so may need to be quite high...
	// },

	// should not need this guy - no formal Output layer
	// {"#Output", emer.Params{
	// 	"Layer.Inhib.Layer.Gi": 1.4, // this turns out to be critical for small output layer
	// }},

	{"#Motor", emer.Params{
		"Layer.Inhib.Layer.Gi": 2.2, // aiming for k = 1 initially
	}},
	{"#Outcome", emer.Params{
		"Layer.Inhib.Layer.Gi": 2.2, // aiming for k = 1 initially
	}},
	{".Back", emer.Params{
		"Prjn.WtScale.Rel": 0.2, // this is generally quite important
	}},
	// // TODO: wrong way to do this; needs concrete projection-specific version; i.e., a "named" prjn?
	// {".Lateral", emer.Params{
	// 	"Prjn.WtScale.Rel": 0.5, // this is generally quite important
	// }},
}

// PlotColorNames are the colors to use (in order) for plotting
// successive lines -- user to customize!
var PlotColorNames = []string{"black", "red", "blue",
	"ForestGreen", "purple", "orange", "brown", "chartreuse",
	"navy", "cyan", "magenta", "tan", "salmon", "yellow4",
	"SkyBlue", "pink"}

// Sim encapsulates the entire simulation model, and we define all the
// functionality as methods on this struct.  This keep all relevant
// state information organized and available without having to pass
// everything around as arguments to methods, and provides the core GUI
// interface (note the view tages for the fields which provide hints to
// how things should be displayed)
// This can be edited directly by the user to access any elements of the simulation.
type Sim struct {
	Net     *leabra.Network `view:"no-inline"`
	ExtReps *etable.Table   `view:"no-inline"`
	EpcLog  *etable.Table   `view:"no-inline"`
	Params  emer.ParamStyle `view:"no-inline"`
	MaxEpcs int             `desc:"maximum number of epochs to run"`
	Epoch   int
	Trial   int

	AlphaCycle int `desc:"0, 1: 0 == 1st, 1 == 2nd alpha-trial of each two-trial sequence"`

	Time leabra.Time

	ViewOn    bool              `desc:"whether to update the network view while running"`
	TrainUpdt leabra.TimeScales `desc:"at what time scale to update the display during training? Anything longer that Epoch updates at Epoch in the model"`
	TestUpdt  leabra.TimeScales `desc:"at what time scale to update the display during training? Anything longer that Epoch updates at Epoch in the model"`

	Plot       bool     `desc:"update the epoch plot while running?"`
	PlotVals   []string `desc:"values to plot in epoch plot"`
	Sequential bool     `desc:"set to true to present items in sequential order"`
	Test       bool     `desc:"set to true to not call learning methods"`

	// statistics
	EpcMotSSE float32 `inactive:"+" desc:"last epoch's total sum squared error - motor layer"`
	EpcOutSSE float32 `inactive:"+" desc:"last epoch's total sum squared error - motor layer"`

	EpcMotAvgSSE float32 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within motor layer)"`
	EpcOutAvgSSE float32 `inactive:"+" desc:"last epoch's average sum squared error (average over trials, and over units within outcome layer)"`

	EpcOutGoalPctErr float32 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance) - compares Outcome to Goal "`
	EpcOutPredPctErr float32 `inactive:"+" desc:"last epoch's percent of trials that had SSE > 0 (subject to .5 unit-wise tolerance) - Outcome layer prediction"`

	EpcOutGoalPctCor float32 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`
	EpcOutPredPctCor float32 `inactive:"+" desc:"last epoch's percent of trials that had SSE == 0 (subject to .5 unit-wise tolerance)"`

	EpcMotCosDiff float32 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`
	EpcOutCosDiff float32 `inactive:"+" desc:"last epoch's average cosine difference for output layer (a normalized error measure, maximum of 1 when the minus phase exactly matches the plus)"`

	// internal state - view:"-"
	GoalSumSSE     float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	GoalSumAvgSSE  float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	GoalSumCosDiff float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`

	MotSumSSE     float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	MotSumAvgSSE  float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	MotSumCosDiff float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`

	OutSumSSE     float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	OutSumAvgSSE  float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`
	OutSumCosDiff float32 `view:"-" inactive:"+" desc:"sum to increment as we go through epoch"`

	OutGoalCntErr int `view:"-" inactive:"+" desc:"sum of errs to increment as we go through epoch"`
	OutPredCntErr int `view:"_" inactive:"+" desc:"sum of prediction errors reflected in Outcome layer as we go through the epoch"`

	Porder     []int       `view:"-" inactive:"+" desc:"permuted pattern order"`
	EpcPlotSvg *svg.Editor `view:"-" desc:"the epoch plot svg editor"`

	NetView *netview.NetView `view:"-" desc:"the network viewer"`

	StopNow bool  `view:"-" desc:"flag to stop running"`
	RndSeed int64 `view:"-" desc:"the current random seed"`
}

// TheSim is the actual instantiation of the simulation and
// tracks all the state values, statistics, etc.
var TheSim Sim

// New creates new blank elements
func (ss *Sim) New() {
	ss.Net = &leabra.Network{}
	ss.ExtReps = &etable.Table{}
	ss.EpcLog = &etable.Table{}
	ss.Params = DefaultParams
	ss.RndSeed = 1

	ss.ViewOn = true
	ss.TrainUpdt = leabra.Cycle
	ss.TestUpdt = leabra.Cycle
}

// Config configures all the elements using the standard functions
func (ss *Sim) Config() {
	ss.ConfigNet()
	ss.OpenExtReps()
	ss.ConfigEpcLog()
}

// Init restarts the run, and initializes everything, including
// network weights and resets the epoch log table
func (ss *Sim) Init() {
	rand.Seed(ss.RndSeed)
	if ss.MaxEpcs == 0 { // allow user override
		ss.MaxEpcs = 500
	}
	ss.Epoch = 0
	ss.Trial = 0
	ss.StopNow = false
	ss.Time.Reset()
	np := ss.ExtReps.NumRows()
	ss.Porder = rand.Perm(np)            // always start with new one so random order is identical
	ss.Net.StyleParams(ss.Params, false) // true) // set msg
	ss.Net.InitWts()
	ss.EpcLog.SetNumRows(0)
	ss.UpdateView()
}

// NewRndSeed gets a new random seed based on current time -- otherwise uses
// the same random seed for every run
func (ss *Sim) NewRndSeed() {
	ss.RndSeed = time.Now().UnixNano()
}

// UpdateView updates the NetView tab visualizing the runnng network
func (ss *Sim) UpdateView() {
	if ss.NetView != nil {
		ss.NetView.Update("Counters:")
	}
}

///////////////////////////////////////////////////////////////
//      Running the Network, starting bottom-up...

// AlphaCyc runs one alpha-trial (100 msec, 4 quarters) of processing
// and corresponds roughly to the original LeabraTrial.
// ApplyInputs() must have already been called prior (e.g., see TrainTrial).
// If learn == true, then DWt and/or WtFmDWt calls are made to update
// weights for learning.
// Handles all NetView updating that is within scope of AlphaCycle.
// But, does NOT handle trial stats nor counter incrementing --
// TrainTrial does that now.
func (ss *Sim) AlphaCyc(train bool) {
	viewUpdt := ss.TrainUpdt
	if !train {
		viewUpdt = ss.TestUpdt
	}
	ss.Net.AlphaCycInit()
	ss.Time.AlphaCycStart()
	for qtr := 0; qtr < 4; qtr++ {
		for cyc := 0; cyc < ss.Time.CycPerQtr; cyc++ {
			// TODO: figure this guy out!!!
			ss.Net.Cycle(&ss.Time)
			ss.Time.CycleInc()
			if ss.ViewOn {
				switch viewUpdt {
				case leabra.Cycle:
					ss.UpdateView()
				case leabra.FastSpike:
					if (cyc+1)%10 == 0 {
						ss.UpdateView()
					}
				}
			}
		}
		ss.Net.QuarterFinal(&ss.Time)
		ss.Time.QuarterInc()
		if ss.ViewOn {
			switch viewUpdt {
			case leabra.Quarter:
				ss.UpdateView()
			case leabra.Phase:
				if qtr >= 2 {
					ss.UpdateView()
				}
			}
		}
	}

	if train {
		ss.Net.DWt()
		ss.Net.WtFmDWt()
		//fmt.Println("Wts should be getting updated.")
	}
	if ss.ViewOn && viewUpdt == leabra.AlphaCycle {
		ss.UpdateView()
	}
}

// ApplyInputs applies input patterns from given row of given table.
// It is good practice to have this be a separate method with
// appropriate args so that it can be used for various different
// contexts (e.g., training, testing, etc.).
// ApplyInputs() must be called BEFORE AlphaCyc()
func (ss *Sim) ApplyInputs(extreps *etable.Table, row int) {
	ss.Net.InitExt() // clear any existing inputs; good practice, cheap

	contextLay := ss.Net.LayerByName("Context").(*leabra.Layer)
	goalLay := ss.Net.LayerByName("Goal").(*leabra.Layer)
	motorLay := ss.Net.LayerByName("Motor").(*leabra.Layer)
	outcomeLay := ss.Net.LayerByName("Outcome").(*leabra.Layer)

	// // OLD WAY
	// contextExtReps := ss.ExtReps.ColByName("Context").(*etensor.Float32)
	// goalExtReps := ss.ExtReps.ColByName("Goal").(*etensor.Float32)
	// motorExtReps := ss.ExtReps.ColByName("Motor").(*etensor.Float32)
	// outcomeExtReps := ss.ExtReps.ColByName("Outcome").(*etensor.Float32)

	// NEW WAY
	contextExtReps := extreps.ColByName(contextLay.Nm).(*etensor.Float32)
	goalExtReps := extreps.ColByName(goalLay.Nm).(*etensor.Float32)
	motorExtReps := extreps.ColByName(motorLay.Nm).(*etensor.Float32)
	outcomeExtReps := extreps.ColByName(outcomeLay.Nm).(*etensor.Float32)

	switch ss.AlphaCycle {
	case 0:
		// Is this where to do this?
		goalLay.SetType(emer.Hidden)
		motorLay.SetType(emer.Hidden)
		outcomeLay.SetType(emer.Target)

		// SubSpace gets the 2D cell at given row in tensor column
		c, _ := contextExtReps.SubSpace(2, []int{row})
		o, _ := outcomeExtReps.SubSpace(2, []int{row})
		contextLay.ApplyExt(c)
		outcomeLay.ApplyExt(o)
		//fmt.Println("AlphaCycle should be 0")
		//fmt.Printf("%d\t%d", ss.AlphaCycle, ss.Trial)
		//fmt.Printf("%d\t%v", row, o)
	case 1:
		goalLay.SetType(emer.Input)
		motorLay.SetType(emer.Target)
		outcomeLay.SetType(emer.Hidden)

		// SubSpace gets the 2D cell at given row in tensor column
		g, _ := goalExtReps.SubSpace(2, []int{row})
		o, _ := outcomeExtReps.SubSpace(2, []int{row})
		g = o
		m, _ := motorExtReps.SubSpace(2, []int{row})

		goalLay.ApplyExt(g)
		motorLay.ApplyExt(m)

		//fmt.Println("AlphaCycle should be 1")
		//fmt.Printf("%d\t%d", ss.AlphaCycle, ss.Trial)
		//fmt.Printf("%v\t%v", row, m)

	default:
		fmt.Println("AlphaCycle appears to be out-of-range")
	}
}

// TrainTrial runs one trial of training (Trial is now an
// environmentally-defined term -- see leabra.TimeScales
// for new, different terminology)
func (ss *Sim) TrainTrial() {
	row := ss.Trial // REMEMBER: two alpha cycles per trial
	if !ss.Sequential {
		row = ss.Porder[ss.Trial]
	}

	//contextLay := ss.Net.LayerByName("Context").(*leabra.Layer)
	//goalLay := ss.Net.LayerByName("Goal").(*leabra.Layer)
	motorLay := ss.Net.LayerByName("Motor").(*leabra.Layer)
	outcomeLay := ss.Net.LayerByName("Outcome").(*leabra.Layer)

	ss.AlphaCycle = 0 // to be safe
	for ss.AlphaCycle < 2 {
		ss.ApplyInputs(ss.ExtReps, row)
		ss.AlphaCyc(true) // train

		// After the 1st AlphaCycle copy Motor and Outcome activation
		// vectors and write to corresponding columns of ExtReps table.
		// (To be used by ApplyInputs() to clamp Goal (emer.Input) and
		// Motor (emer.Target) in the 2nd AlphaCycle.
		var msz, osz int
		if ss.AlphaCycle == 0 {
			mav, errm := motorLay.UnitVals("ActP") // mav returned of type []float32
			//mav, _ := motorLay.UnitVals("ActP") // mav returned of type []float32
			msz = len(mav)

			tnsr := ss.ExtReps.ColByName("Motor")
			_, cells := tnsr.RowCellSize()
			stidx := row * cells
			if errm == nil {
				for i := range mav[0:] {
					tnsr.SetFloat1D(stidx+i, float64(mav[i]))
					//ss.ExtReps.ColByName("Motor").SetFloat1D(stidx+i, float64(mav[i]))
				}
			}
			// // less safe version...
			// for i := range mav[0:] {
			// 	ss.ExtReps.ColByName("Motor").SetFloat1D(stidx+i, float64(mav[i]))
			// }

			oav, err := outcomeLay.UnitVals("ActP")
			//oav, _ := outcomeLay.UnitVals("ActP")
			osz = len(oav)

			tsr := ss.ExtReps.ColByName("Outcome")
			_, cels := tsr.RowCellSize()
			sidx := row * cels
			if err == nil {
				for j := range oav[0:] {
					tsr.SetFloat1D(sidx+j, float64(oav[j]))
					//ss.ExtReps.ColByName("Outcome").SetFloat1D(sidx+j, float64(oav[j]))
				}
			}
			// // less safe version...
			// for j := range oav[0:] {
			// 	ss.ExtReps.ColByName("Outcome").SetFloat1D(sidx+j, float64(oav[j]))
			// }
		}
		if ss.AlphaCycle >= 1 {
			// Reset ExtReps Motor and Outcome activation vectors
			for j := 0; j < msz; j++ {
				ss.ExtReps.ColByName("Motor").SetFloat1D(row+j, float64(0))
			}
			for j := 0; j < osz; j++ {
				ss.ExtReps.ColByName("Outcome").SetFloat1D(row+j, float64(0))
			}
			break
		}
		ss.TrialStats(true) // accumulate // TODO: figure out stat tracking - trial-level vs. alpha-level, etc.
		ss.AlphaCycle++     // TODO: how to make this display as it changes?
	}
	//ss.AlphaCycle = 0 // reset for next time through to be sure

	ss.TrialStats(true) // accumulate // TODO: figure out stat tracking - trial-level vs. alpha-level, etc.

	// To allow for interactive single-step running, all of the
	// higher temporal scales must be incorporated into the trial
	// level run method.
	// This is a good general principle for even more complex
	// environments:
	// there should be a single method call that gets the next "step"
	// of the environment, and all the higher levels of temporal
	// structure sould all be properl updated thourgh this one lowest-
	// level method call.

	ss.Trial++
	nr := ss.ExtReps.NumRows()
	if ss.Trial >= nr {
		ss.LogEpoch()
		if ss.Plot {
			ss.PlotEpcLog()
		}
		ss.Trial = 0
		ss.Epoch++
		erand.PermuteInts(ss.Porder)
		if ss.ViewOn && ss.TrainUpdt > leabra.AlphaCycle {
			ss.UpdateView()
		}
	}
}

// TrialStats computes the trial-level statistics and adds them to
// the epoch accumulators if accum is true.
// Note that we're accumulating stats here on the Sim side so the
// core algorithmic side remains as simple as possible, and doesn't
// need to worry about different time-scales over which stats could
// be accumulated, etc.
func (ss *Sim) TrialStats(accum bool) (gsse, msse, osse, gavgsse, mavgsse, oavgsse, motcosdiff, outcosdiff float32) {
	goalLay := ss.Net.LayerByName("Goal").(*leabra.Layer)
	motorLay := ss.Net.LayerByName("Motor").(*leabra.Layer)
	outcomeLay := ss.Net.LayerByName("Outcome").(*leabra.Layer)

	switch ss.AlphaCycle {
	// TODO: NOTE: figure out and/or keep track of what/if accumulating only every other trial does to averages, etc.
	case 0:
		gsse, gavgsse = goalLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
		//msse, mavgsse = motorLay.MSE(0.5)   // 0.5 = per-unit tolerance -- right side of .5
		osse, oavgsse = outcomeLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5

		//goalcosdiff = goalLay.CosDiff.Cos
		outcosdiff = outcomeLay.CosDiff.Cos

		// sseg, errg := goalLay.UnitVals("ActM")
		// sseo, erro := outcomeLay.UnitVals("ActM")
		// if errg != nil && erro != nil {
		// 	// take difference of sseg - sseo and calculate GoalCntErr
		// }
		if accum {
			ss.OutSumSSE += osse
			ss.OutSumAvgSSE += oavgsse
			ss.OutSumCosDiff += outcosdiff

			if osse != 0 {
				ss.OutPredCntErr++
			}
		}

		tol := float32(0.5)
		nng := len(goalLay.Neurons)
		nno := len(outcomeLay.Neurons)
		if nng != nno {
			fmt.Println("Number of neuron-units does not match between Goal and Outcome layers")
		}
		oge := float32(0)
		for ni := range goalLay.Neurons {
			ng := &goalLay.Neurons[ni]
			no := &outcomeLay.Neurons[ni]
			d := ng.ActM - no.ActM
			if math32.Abs(d) < tol {
				continue
			}
			oge += d * d
		}
		if oge != 0 {
			ss.OutGoalCntErr++
		}

	case 1:
		msse, mavgsse = motorLay.MSE(0.5) // 0.5 = per-unit tolerance -- right side of .5
		motcosdiff = motorLay.CosDiff.Cos
		if accum {
			ss.MotSumSSE += msse
			ss.MotSumAvgSSE += mavgsse
			ss.MotSumCosDiff += motcosdiff
		}

	default:
		fmt.Println("TrialStats says AlphaCycle appears to be out-of-range")
	}
	return
}

// EpochInc increments counters after one epoch of processing and updates a new random
// order of permuted inputs for the next epoch
func (ss *Sim) EpochInc() {
	ss.Trial = 0
	ss.Epoch++
	erand.PermuteInts(ss.Porder)
}

// LogEpoch adds data from current epoch to the EpochLog table
// -- computes epoch averages prior to logging.
// Epoch counter is assumed to not have yet been incremented.
func (ss *Sim) LogEpoch() {
	ss.EpcLog.SetNumRows(ss.Epoch + 1)
	contextLay := ss.Net.LayerByName("Context").(*leabra.Layer)
	goalLay := ss.Net.LayerByName("Goal").(*leabra.Layer)
	motorLay := ss.Net.LayerByName("Motor").(*leabra.Layer)
	outcomeLay := ss.Net.LayerByName("Outcome").(*leabra.Layer)

	np := float32(ss.ExtReps.NumRows())
	//ss.EpcGoalSSE = ss.GoalSumSSE / np
	ss.EpcMotSSE = ss.MotSumSSE / np
	ss.EpcOutSSE = ss.OutSumSSE / np

	//ss.GoalSumSSE = 0
	ss.MotSumSSE = 0
	ss.OutSumSSE = 0

	//ss.EpcGoalAvgSSE = ss.GoalSumAvgSSE / np
	ss.EpcMotAvgSSE = ss.MotSumAvgSSE / np
	ss.EpcOutAvgSSE = ss.OutSumAvgSSE / np

	//ss.GoalSumAvgSSE = 0
	ss.MotSumAvgSSE = 0
	ss.OutSumAvgSSE = 0

	ss.EpcOutGoalPctErr = float32(ss.OutGoalCntErr) / np
	ss.EpcOutPredPctErr = float32(ss.OutPredCntErr) / np

	ss.OutGoalCntErr = 0
	ss.OutPredCntErr = 0

	ss.EpcOutGoalPctCor = 1 - ss.EpcOutGoalPctErr
	ss.EpcOutPredPctCor = 1 - ss.EpcOutPredPctErr

	ss.EpcMotCosDiff = ss.MotSumCosDiff / np
	ss.EpcOutCosDiff = ss.OutSumCosDiff / np

	ss.MotSumCosDiff = 0
	ss.OutSumCosDiff = 0

	epc := ss.Epoch

	ss.EpcLog.ColByName("Epoch").SetFloat1D(epc, float64(epc))
	ss.EpcLog.ColByName("MotSSE").SetFloat1D(epc, float64(ss.EpcMotSSE))
	ss.EpcLog.ColByName("OutSSE").SetFloat1D(epc, float64(ss.EpcOutSSE))

	ss.EpcLog.ColByName("MotAvgSSE").SetFloat1D(epc, float64(ss.EpcMotAvgSSE))
	ss.EpcLog.ColByName("OutAvgSSE").SetFloat1D(epc, float64(ss.EpcOutAvgSSE))

	ss.EpcLog.ColByName("OutGoalPctErr").SetFloat1D(epc, float64(ss.EpcOutGoalPctErr))
	ss.EpcLog.ColByName("OutPredPctErr").SetFloat1D(epc, float64(ss.EpcOutPredPctErr))

	ss.EpcLog.ColByName("OutGoalPctCor").SetFloat1D(epc, float64(ss.EpcOutGoalPctCor))
	ss.EpcLog.ColByName("OutPredPctCor").SetFloat1D(epc, float64(ss.EpcOutPredPctCor))

	ss.EpcLog.ColByName("MotCosDiff").SetFloat1D(epc, float64(ss.EpcMotCosDiff))
	ss.EpcLog.ColByName("OutCosDiff").SetFloat1D(epc, float64(ss.EpcOutCosDiff))

	//ss.EpcLog.ColByName("ContextActAvg").SetFloat1D(epc, float64(contextLay.Pools[0].ActAvg.ActPAvgEff))
	//ss.EpcLog.ColByName("GoalActAvg").SetFloat1D(epc, float64(goalLay.Pools[0].ActAvg.ActPAvgEff))
	//ss.EpcLog.ColByName("MotorActAvg").SetFloat1D(epc, float64(motorLay.Pools[0].ActAvg.ActPAvgEff))
	//ss.EpcLog.ColByName("OutActAvg").SetFloat1D(epc, float64(outcomeLay.Pools[0].ActAvg.ActPAvgEff))
	ss.EpcLog.ColByName("ContextActAvg").SetFloat1D(epc, float64(contextLay.Pools[0].ActAvg.ActMAvg))
	ss.EpcLog.ColByName("GoalActAvg").SetFloat1D(epc, float64(goalLay.Pools[0].ActAvg.ActMAvg))
	ss.EpcLog.ColByName("MotorActAvg").SetFloat1D(epc, float64(motorLay.Pools[0].ActAvg.ActMAvg))
	ss.EpcLog.ColByName("OutActAvg").SetFloat1D(epc, float64(outcomeLay.Pools[0].ActAvg.ActMAvg))

	ss.EpcLog.ColByName("OutGoalCntErr").SetFloat1D(epc, float64(ss.OutGoalCntErr))
	ss.EpcLog.ColByName("OutPredCntErr").SetFloat1D(epc, float64(ss.OutPredCntErr))
}

// TrainEpoch runs one full epoch at a time; when stopped mid-epoch finishes current epoch
func (ss *Sim) TrainEpoch() {
	curEpc := ss.Epoch
	for {
		ss.TrainTrial()
		//ss.TrialStats(!ss.Test) // accumulate if not doing testing
		//ss.TrialInc()           // does LogEpoch, EpochInc automatically
		if ss.StopNow || ss.Epoch > curEpc {
			break
		}
	}
}

// Train runs the full training from this point onward
func (ss *Sim) Train() {
	ss.StopNow = false
	stEpc := ss.Epoch
	tmr := timer.Time{}
	tmr.Start()
	for {
		ss.TrainTrial()
		if ss.StopNow || ss.Epoch >= ss.MaxEpcs {
			break
		}
	}
	tmr.Stop()
	epcs := ss.Epoch - stEpc
	fmt.Printf("Took %6g secs for %v epochs, avg per epc: %6g\n", tmr.TotalSecs(), epcs, tmr.TotalSecs()/float64(epcs))
}

// Stop tells the sim to stop running
func (ss *Sim) Stop() {
	ss.StopNow = true
}

///////////////////////////////////////////////////////////
// Testing

// TestTrial runs one trial of testing -- always sequentially
// presented inputs
func (ss *Sim) TestTrial() {
	//TODO: ...
}

// TestAll runs through the full set of testing items
func (ss *Sim) TestAll() {
	nr := ss.ExtReps.NumRows()
	ss.Trial = 0
	for trl := 0; trl < nr; trl++ {
		ss.TestTrial()
	}
}

//////////////////////////////////////////////////////////
// Config methods

// ConfigNet sets up the network prior to running
func (ss *Sim) ConfigNet() {
	net := ss.Net
	net.InitName(net, "GoalGuyNet")
	contextLay := net.AddLayer2D("Context", 5, 5, emer.Input)
	goalLay := net.AddLayer2D("Goal", 5, 5, emer.Hidden)
	motorLay := net.AddLayer2D("Motor", 5, 5, emer.Hidden)
	outcomeLay := net.AddLayer2D("Outcome", 5, 5, emer.Target)

	// BELOW for reference only:
	//hid2Lay := net.AddLayer4D("Hidden2", 2, 4, 3, 2, emer.Hidden) // outerY, X, innerY, X
	// AND: use this to position layers relative to each other
	// default is Above, YAlign = Front, XAligh = Center
	//hid2Lay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Hidden1", YAlign: relpos.Front, Space: 2})

	contextLay.SetRelPos(relpos.Rel{Rel: relpos.Above, Other: "Motor", YAlign: relpos.Front, Space: 2})
	outcomeLay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Motor", YAlign: relpos.Front, Space: 2})
	goalLay.SetRelPos(relpos.Rel{Rel: relpos.RightOf, Other: "Context", YAlign: relpos.Front, Space: 2})

	net.ConnectLayers(contextLay, goalLay, prjn.NewOneToOne(), emer.Forward)
	//net.ConnectLayers(goalLay, motorLay, prjn.NewOneToOne(), emer.Forward)
	net.ConnectLayers(goalLay, motorLay, prjn.NewFull(), emer.Forward)
	net.ConnectLayers(motorLay, outcomeLay, prjn.NewFull(), emer.Forward)
	// Trying weaker inputs to Outcome layer - did NOT seem to help...
	//net.ConnectLayers(motorLay, outcomeLay, prjn.NewFull(), emer.Lateral)

	net.ConnectLayers(outcomeLay, motorLay, prjn.NewFull(), emer.Back)
	//net.ConnectLayers(motorLay, goalLay, prjn.NewFull(), emer.Back)

	// if Thread {
	// 	motorLay.SetThread(1)
	// 	outcomeLay.SetThread(1)
	// }

	net.Defaults()
	net.StyleParams(ss.Params, true) // set msg
	net.Build()
	net.InitWts()
}

// ConfigExtReps creates a new version of the ExtReps table and writes it to
// permanent storage as goal_guy_0_5X5_25_gen.dat file in the local directory
func (ss *Sim) ConfigExtReps() {
	et := ss.ExtReps
	et.SetFromSchema(etable.Schema{
		{"Name", etensor.STRING, nil, nil},
		{"Context", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Goal", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Motor", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
		{"Outcome", etensor.FLOAT32, []int{5, 5}, []string{"Y", "X"}},
	}, 25) // 250

	patgen.PermutedBinaryRows(et.Cols[1], 3, 1, 0)
	patgen.PermutedBinaryRows(et.Cols[2], 0, 0, 0)
	patgen.PermutedBinaryRows(et.Cols[3], 0, 0, 0)
	patgen.PermutedBinaryRows(et.Cols[4], 3, 1, 0)
	et.SaveCSV("goal-guy-0-5x5-25-gen.dat", ',', true)
}

// OpenExtReps opens an existing (permanent) CSV version of the ExtReps file
func (ss *Sim) OpenExtReps() {
	et := ss.ExtReps
	err := et.OpenCSV("goal-guy-0-5x5-25-gen.dat", '\t')
	if err != nil {
		log.Println(err)
	}
}

// ConfigEpcLog sets up the EpcLog table
func (ss *Sim) ConfigEpcLog() {
	et := ss.EpcLog
	et.SetFromSchema(etable.Schema{
		{"Epoch", etensor.INT64, nil, nil},
		{"MotSSE", etensor.FLOAT32, nil, nil},
		{"OutSSE", etensor.FLOAT32, nil, nil},

		{"MotAvgSSE", etensor.FLOAT32, nil, nil},
		{"OutAvgSSE", etensor.FLOAT32, nil, nil},

		{"OutGoalPctErr", etensor.FLOAT32, nil, nil},
		{"OutPredPctErr", etensor.FLOAT32, nil, nil},

		{"OutGoalPctCor", etensor.FLOAT32, nil, nil},
		{"OutPredPctCor", etensor.FLOAT32, nil, nil},

		{"MotCosDiff", etensor.FLOAT32, nil, nil},
		{"OutCosDiff", etensor.FLOAT32, nil, nil},

		{"ContextActAvg", etensor.FLOAT32, nil, nil},
		{"GoalActAvg", etensor.FLOAT32, nil, nil},
		{"MotorActAvg", etensor.FLOAT32, nil, nil},
		{"OutActAvg", etensor.FLOAT32, nil, nil},

		{"OutPredCntErr", etensor.FLOAT32, nil, nil},
		{"OutGoalCntErr", etensor.FLOAT32, nil, nil},
	}, 0)
	//ss.PlotVals = []string{"OutSSE", "Out Goal Pct Err"}
	ss.PlotVals = []string{"OutCosDiff", "MotCosDiff", "OutGoalPctErr"}
	ss.Plot = true
}

// PlotEpcLog plots given epoch log using PlotVals Y axis
// columns into EpcPlotSvg
func (ss *Sim) PlotEpcLog() *plot.Plot {
	if !ss.EpcPlotSvg.IsVisible() {
		return nil
	}
	et := ss.EpcLog
	plt, _ := plot.New() // todo: keep around?
	plt.Title.Text = "Goal Guy Epoch Log"
	plt.X.Label.Text = "Epoch"
	plt.Y.Label.Text = "Y"

	const lineWidth = 1

	for i, cl := range ss.PlotVals {
		xy, _ := eplot.NewTableXYNames(et, "Epoch", cl)
		l, _ := plotter.NewLine(xy)
		l.LineStyle.Width = vg.Points(lineWidth)
		clr, _ := gi.ColorFromString(PlotColorNames[i%len(PlotColorNames)], nil)
		l.LineStyle.Color = clr
		plt.Add(l)
		plt.Legend.Add(cl, l)
	}
	plt.Legend.Top = true
	//eplot.PlotViewSVG(plt, ss.EpcPlotSvg, 5, 5, 2)
	eplot.PlotViewSVG(plt, ss.EpcPlotSvg, 5)
	return plt
}

// SaveEpcPlot plots given epoch log using PlotVals Y axis columns and saves to .svg file
func (ss *Sim) SaveEpcPlot(fname string) {
	plt := ss.PlotEpcLog()
	plt.Save(5, 5, fname)
}

// ConfigGui configures the GoGi gui interface for this simulation,
func (ss *Sim) ConfigGui() *gi.Window {
	width := 1600
	height := 1200

	gi.SetAppName("goal-guy-0")
	gi.SetAppAbout(`This demonstrates learning of basic goal-directed behavior. See <a href="https://github.com/emer/emergent">emergent on GitHub</a>.</p>`)

	plot.DefaultFont = "Helvetica"

	win := gi.NewWindow2D("goal-guy-0", "Goal Guy Phase 0", width, height, true)

	vp := win.WinViewport2D()
	updt := vp.UpdateStart()

	mfr := win.SetMainFrame()

	tbar := gi.AddNewToolBar(mfr, "tbar")
	tbar.SetStretchMaxWidth()

	split := gi.AddNewSplitView(mfr, "split")
	split.Dim = gi.X
	// split.SetProp("horizontal-align", "center")
	// split.SetProp("margin", 2.0) // raw numbers = px = 96 dpi pixels
	split.SetStretchMaxWidth()
	split.SetStretchMaxHeight()

	sv := giv.AddNewStructView(split, "sv")
	sv.SetStruct(ss, nil)
	// sv.SetStretchMaxWidth()
	// sv.SetStretchMaxHeight()

	tv := gi.AddNewTabView(split, "tv")

	nv := tv.AddNewTab(netview.KiT_NetView, "NetView").(*netview.NetView)
	nv.SetStretchMaxWidth()
	nv.SetStretchMaxHeight()
	nv.Var = "Act"
	nv.SetNet(ss.Net)
	ss.NetView = nv

	svge := tv.AddNewTab(svg.KiT_Editor, "Epc Plot").(*svg.Editor)
	svge.InitScale()
	svge.Fill = true
	svge.SetProp("background-color", "white")
	svge.SetProp("width", units.NewValue(float32(width/2), units.Px))
	svge.SetProp("height", units.NewValue(float32(height-100), units.Px))
	svge.SetStretchMaxWidth()
	svge.SetStretchMaxHeight()
	ss.EpcPlotSvg = svge

	split.SetSplits(.3, .7)

	tbar.AddAction(gi.ActOpts{Label: "Init", Icon: "update"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Init()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Train", Icon: "run"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			go ss.Train()
		})

	tbar.AddAction(gi.ActOpts{Label: "Stop", Icon: "stop"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Stop()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("text")
	tbar.AddSeparator("text")

	tbar.AddAction(gi.ActOpts{Label: "Step Trial", Icon: "step-fwd"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TrainTrial()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Step Epoch", Icon: "fast-fwd"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.TrainEpoch()
			vp.FullRender2DTree()
		})

	// tbar.AddSep("file")
	tbar.AddSeparator("text")
	tbar.AddSeparator("text")

	tbar.AddAction(gi.ActOpts{Label: "Test Trial", Icon: "step-fwd"}, win.This(),
		func(rev, send ki.Ki, sig int64, data interface{}) {
			ss.TestTrial()
			vp.FullRender2DTree()
		})

	tbar.AddAction(gi.ActOpts{Label: "Test All", Icon: "step-fwd"}, win.This(),
		func(rev, send ki.Ki, sig int64, data interface{}) {
			ss.TestAll()
			vp.FullRender2DTree()
		})

	tbar.AddSeparator("text")
	tbar.AddSeparator("text")

	tbar.AddAction(gi.ActOpts{Label: "Epoch Plot", Icon: "update"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.PlotEpcLog()
		})

	tbar.AddSeparator("text")
	tbar.AddSeparator("text")

	tbar.AddAction(gi.ActOpts{Label: "Save Wts", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.Net.SaveWtsJSON("goal_guy_0_net_trained.wts") // todo: call method to prompt
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Log", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.EpcLog.SaveCSV("goal_guy_0_epc.dat", ',', true)
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Plot", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.SaveEpcPlot("goal_guy_0_cur_epc_plot.svg")
		})

	tbar.AddAction(gi.ActOpts{Label: "Save Params", Icon: "file-save"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			// todo: need save / load methods for these
			// ss.EpcLog.SaveCSV("goal_guy_0_params.dat", ',', true)
		})

	tbar.AddSeparator("text")
	tbar.AddSeparator("text")

	tbar.AddAction(gi.ActOpts{Label: "New Seed", Icon: "new"}, win.This(),
		func(recv, send ki.Ki, sig int64, data interface{}) {
			ss.NewRndSeed()
		})

	vp.UpdateEndNoSig(updt)

	// main menu
	appnm := gi.AppName()
	mmen := win.MainMenu
	mmen.ConfigMenus([]string{appnm, "File", "Edit", "Window"})

	amen := win.MainMenu.ChildByName(appnm, 0).(*gi.Action)
	amen.Menu.AddAppMenu(win)

	emen := win.MainMenu.ChildByName("Edit", 1).(*gi.Action)
	emen.Menu.AddCopyCutPaste(win)

	// note: Command in shortcuts is automatically translated into Control for
	// Linux, Windows or Meta for MacOS
	// fmen := win.MainMenu.ChildByName("File", 0).(*gi.Action)
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Open", Shortcut: "Command+O"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		FileViewOpenSVG(vp)
	// 	})
	// fmen.Menu.AddSeparator("csep")
	// fmen.Menu.AddAction(gi.ActOpts{Label: "Close Window", Shortcut: "Command+W"},
	// 	win.This(), func(recv, send ki.Ki, sig int64, data interface{}) {
	// 		win.Close()
	// 	})

	win.SetCloseCleanFunc(func(w *gi.Window) {
		go gi.Quit() // once main window is closed, quit
	})

	win.MainMenuUpdated()
	return win
}

func mainrun() {
	// gi3d.Update3DTrace = true
	// gi.Update2DTrace = true
	// gi.Render2DTrace = true

	// todo: args
	TheSim.New()

	// Run below only to generate ExtReps table to hold externally clamped representations
	// else comment out...
	TheSim.ConfigExtReps()

	TheSim.Config()
	TheSim.Init()
	win := TheSim.ConfigGui()
	win.StartEventLoop()

}
