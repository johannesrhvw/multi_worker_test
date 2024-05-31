import logging
import time

from configuration.config import Config

from .paddle_lookup import PADDLE_PINS


class DummySorter:
    def __init__(self, config: Config) -> None:
        """
        Sorter controlling a festo cpx ip controller
        connected to multiple valve terminals.
        Uses a paddle lookup file to create compatible signals from a received
        integer list indicating the number of iterations until a valve with the
        corresponding index has to be activated.
        The class calling run() should implement a loop.

        Args:
            clock_period (float): Estimate "clock" time for the execution.
            ip (str): IP address of the festo cpx ip controller.

        Raises:
            ValueError: If the desired clock time is longer then 1second.
        """
        sort_config = config.general_config.sorter_config
        self.logger = logging.getLogger(__name__)
        self.ip = sort_config.ip_address
        self.output_length = len(PADDLE_PINS)
        self.active_valve_time = sort_config.active_valve_time
        if sort_config.sort_interval > self.active_valve_time:
            raise ValueError(
                f"Clock period {sort_config.sort_interval}s exceeds valve " f"active time {self.active_valve_time}s."
            )
        self.clk_period = sort_config.sort_interval
        # this value is used for counting down over clock_period periods to
        # achive desired active paddle time
        self.active_count = 1  # int(self.active_valve_time / self.clk_period)
        # this list holds all integer values to be counted down for each paddle
        self.sortlist: list[list[int]] = [[] for _ in range(self.output_length)]

    def run(self, signal: list[int] | None) -> None:
        """
        Creates a compatible signal for the cpx ip from the provided integer
        list.
        Args:
            signal (list[int] | None): List containing counter values
            indicating when a paddle with the corresponding value index
            must be activated.
        """
        if signal is not None:
            for i, value in enumerate(signal):
                if isinstance(value, int):
                    wait_count = value
                    if value > 0:
                        # add the active count to the count until activation
                        wait_count += self.active_count
                        # add new values to the sortlist
                        self.sortlist[i].append(wait_count)
        # create a signal from the sortlist by counting down the integer values
        bool_signal = self._signal_from_sortlist()
        self._send_signal(bool_signal)
        time.sleep(self.clk_period)

    def _signal_from_sortlist(self) -> list[bool]:
        """
        Create a bool signal from the integer list containing
        counter values for each paddle index.

        Returns:
            list[bool]: List containing the signal for each paddle.
        """
        # default signal deactivates all paddles
        # output length is equal to the total number of paddles to
        # be controlled
        signal = [False] * self.output_length
        # each paddle has a list of counter values to be decreased by -1
        # in each iteration
        for i, paddle in enumerate(self.sortlist):
            for j, number in enumerate(paddle):
                # if a value in the list of counters for paddle i
                # is below 0 this state should not be reachable
                if number < 0:
                    paddle[j] = 0
                    self.logger.error(f"Number {number} is negative.")
                # is greater then the active count the value will be decreased
                # and the corresponding signal stays low/False
                if number > self.active_count:
                    paddle[j] -= 1
                # is lower then the active count the paddle has reached active
                # phase, the signal is high, until the value reaches 0
                if number <= self.active_count and number > 0:
                    signal[i] = True
                    paddle[j] -= 1
                # if the active count is over, the paddle
                # will be reset to low/False
                if number == 0 and len(paddle) > 1:
                    paddle.pop(j)
        return signal

    def _send_signal(self, signal: list[bool]) -> None:
        if any(value is True for value in signal):
            self.logger.debug(f"Send signal{signal} to cpx ip.")
