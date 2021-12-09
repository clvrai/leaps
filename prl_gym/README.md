# Neural Network Policy:
   - agent: karel_state -> NN -> action
   - karel_world: action -> next_karel_state, reward(currently sparse)

# Programmatic Policy:
   ## Execution environment:
   - agent: (karel_state, program_counter) -> Program -> action
   - karel_world: action -> (next_karel_state, reward(currently sparse))

   ## Program Environment 1:
   - agent: <start> -> NN -> program
   - environment: program(action) -> (next_state(currently None), reward)
        
   ## Program Environment 2:
   - agent: <start> -> NN -> program
   - environment: program(action) -> (next_state(currently None), reward (task done or not))

