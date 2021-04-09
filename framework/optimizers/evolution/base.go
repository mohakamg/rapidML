package evolution

type Metrics map[string]float32

type evolutionState struct {
	currentGeneration int
	currentPopulation chan interface{}
	reproductionPool chan interface{}
}

type EvolutionaryParams struct {
	NumGenerations int
	PopulationSize int
	NumSpecies int

	evolutionState
}

type Evolution interface {
	Initialize(chan interface{}) error
	EvaluateCandidate(interface{}) (error, Metrics)
	Select() error
	Crossover() error
	Mutate() error
	Speciate() error

	run() error
}

func (state *EvolutionaryParams) Initialize(chan interface{}) error                   { return nil }
func (state *EvolutionaryParams) EvaluateCandidate(interface{}) (error, Metrics) { return nil, nil }
func (state *EvolutionaryParams) Select() error                       { return nil }
func (state *EvolutionaryParams) Crossover() error                    { return nil }
func (state *EvolutionaryParams) Mutate() error                       { return nil }
func (state *EvolutionaryParams) Speciate() error                     { return nil }

func (state *EvolutionaryParams) run() (err error) {

	population := make(chan interface{})
	if err = state.Initialize(population); err != nil {
		return
	}






}