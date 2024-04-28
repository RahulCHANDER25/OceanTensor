SRC	=	src/main.cpp 	\
		OceanTensor/Metadata.cpp 	\
		OceanTensor/operations/tensorOp.cpp

OBJ	=	$(SRC:.cpp=.o)

CC	:=	g++
RM	:=	rm -rf
CPPFLAGS	=	-iquote OceanTensor/
CFLAGS	=	-Wall -g3

NAME	=	OceanTensor

all: $(NAME)

$(NAME):	$(OBJ)
	$(CC) -o $(NAME) $(OBJ) $(CFLAGS)

clean:
	$(RM) $(OBJ)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: fclean re all clean
