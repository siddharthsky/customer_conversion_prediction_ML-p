import sys
sys.path.append(os.getcwd())

def error_message_details(er,er_detail:sys):
    _,_,exc_tb=er_detail.exc_info()
    file_name=exc_tb.tb_frame.f_code.co_filename
    error_message="Error occured in python script name [{0}]. line number [{1}]. error message [{2}]".format(
    file_name,exc_tb.tb_lineno,str(er))
    
    return(error_message)


class CustomException(Exception):
    def __init__(self,error_message,er_detail:sys):
        super().__init__(error_message)
        self.error_message=error_message_details(error_message,er_detail=er_detail)

    def __str__(self):
        return self.error_message