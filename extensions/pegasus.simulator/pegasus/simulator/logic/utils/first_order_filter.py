"""
| File: first_order_filter.py
| Author: Tang Longbin (longbin@u.nus.edu)
| Description: File that implements a first order filter, which allows different acceleration and deceleration time constants.
| License: BSD-3-Clause. Copyright (c) 2024, Tang Longbin. All rights reserved.
"""
import math

class FirstOrderFilter:
    """
    This class can be used to apply a first order filter on a signal.
    It allows different acceleration and deceleration time constants.

    Short review of discrete time implementation of first order system:
    Laplace:
        X(s)/U(s) = 1/(tau*s + 1)
    continuous time system:
        dx(t) = (-1/tau)*x(t) + (1/tau)*u(t)
    discretized system (ZoH):
        x(k+1) = exp(samplingTime*(-1/tau))*x(k) + (1 - exp(samplingTime*(-1/tau))) * u(k)
    """

    def __init__(self, time_constant_up, time_constant_down, initial_state):
        self._time_constant_up = time_constant_up
        self._time_constant_down = time_constant_down
        assert time_constant_up > 0 and time_constant_down > 0
        self._previous_state = initial_state

    def update(self, input_state, sampling_time):
        """
        This method will apply a first order filter on the input_state.
        """
        if input_state > self._previous_state:
            # Calculate the output_state if accelerating.
            alpha_up = math.exp(-sampling_time / self._time_constant_up)
            # x(k+1) = Ad*x(k) + Bd*u(k)
            output_state = alpha_up * self._previous_state + (1 - alpha_up) * input_state
        else:
            # Calculate the output_state if decelerating.
            alpha_down = math.exp(-sampling_time / self._time_constant_down)
            output_state = alpha_down * self._previous_state + (1 - alpha_down) * input_state

        self._previous_state = output_state
        return output_state
    
if __name__ == "__main__":
    """
    example of using the first order filter.
    """
    time_constant_up = 0.1
    time_constant_down = 0.2
    sampling_time = 0.01
    intial_state = 0
    filter = FirstOrderFilter(time_constant_up, time_constant_down, intial_state)
    
    states = [0]
    for i in range(100):
        # simulate a step acceleration
        state = filter.update(1, sampling_time)
        states.append(state)

    for i in range(100):
        # simulate a step deceleration
        state = filter.update(0, sampling_time)
        states.append(state)
    
    '''
    Plot the response of the first order filter.
    '''
    import matplotlib.pyplot as plt
    time = [i * sampling_time for i in range(len(states))]
    plt.plot(time, states)
    # Find the first state == 0.1 and state == 0.67
    first_state_1 = next((i for i, state in enumerate(states) if state >= 0.1), None)
    first_state_67 = next((i for i, state in enumerate(states) if state >= 0.67), None)
    
    if first_state_1 is not None and first_state_67 is not None:
        plt.scatter(time[first_state_1], states[first_state_1], color='red')
        plt.scatter(time[first_state_67], states[first_state_67], color='red')
        plt.annotate(f'0.1 at {time[first_state_1]:.2f}s', (time[first_state_1], states[first_state_1]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'0.67 at {time[first_state_67]:.2f}s', (time[first_state_67], states[first_state_67]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'tau: {time[first_state_67] - time[first_state_1]:.2f}s', 
                     ((time[first_state_1] + time[first_state_67]) / 2, (states[first_state_1] + states[first_state_67]) / 2), 
                     textcoords="offset points", xytext=(0,10), ha='center')

    # Find the last state == 1 and state == 0.33
    last_state_9 = next((i for i, state in reversed(list(enumerate(states))) if state >= 0.9), None)
    last_state_33 = next((i for i, state in reversed(list(enumerate(states))) if state >= 0.33), None)
    
    if last_state_9 is not None and last_state_33 is not None:
        plt.scatter(time[last_state_9], states[last_state_9], color='blue')
        plt.scatter(time[last_state_33], states[last_state_33], color='blue')
        plt.annotate(f'0.9 at {time[last_state_9]:.2f}s', (time[last_state_9], states[last_state_9]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'0.33 at {time[last_state_33]:.2f}s', (time[last_state_33], states[last_state_33]), textcoords="offset points", xytext=(0,10), ha='center')
        plt.annotate(f'tau: {time[last_state_33] - time[last_state_9]:.2f}s', 
                     ((time[last_state_9] + time[last_state_33]) / 2, (states[last_state_9] + states[last_state_33]) / 2), 
                     textcoords="offset points", xytext=(0,10), ha='center')
    
    plt.xlabel('Time (s)')
    plt.ylabel('State')
    plt.title('First Order Filter Response')
    plt.grid()
    plt.show()
