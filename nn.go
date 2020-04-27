package main

import (
	"gorgonia.org/gorgonia"
	"gorgonia.org/tensor"
)

type nn struct {
	g       *gorgonia.ExprGraph
	w0      *gorgonia.Node
	w1      *gorgonia.Node
	out     *gorgonia.Node
	predVal gorgonia.Value
}

func newNN(g *gorgonia.ExprGraph) *nn {
	w0 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(10, 500),
		gorgonia.WithName("w0"),
		gorgonia.WithInit(gorgonia.Zeroes()))

	w1 := gorgonia.NewMatrix(
		g,
		tensor.Float64,
		gorgonia.WithShape(500, 4),
		gorgonia.WithName("w1"),
		gorgonia.WithInit(gorgonia.GlorotU(1)))

	return &nn{
		g:  g,
		w0: w0,
		w1: w1,
	}
}

func (m *nn) learnables() gorgonia.Nodes {
	return gorgonia.Nodes{m.w0, m.w1}
}

func (m *nn) fwd(x *gorgonia.Node) error {
	l0 := x
	l0dot := gorgonia.Must(gorgonia.Mul(l0, m.w0))
	l1 := gorgonia.Must(gorgonia.Rectify(l0dot))
	l1dot := gorgonia.Must(gorgonia.Mul(l1, m.w1))
	l2 := gorgonia.Must(gorgonia.Sigmoid(l1dot))
	m.out = l2
	gorgonia.Read(l1, &m.predVal)
	return nil
}
