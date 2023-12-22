class InvalidArgumentException(Exception):
    def __init__(self, arg) -> None:
        super().__init__()
        self.arg = arg
    
    def __str__(self) -> str:
        return f"{self.arg} is invalid argument."