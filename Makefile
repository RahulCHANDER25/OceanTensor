SRC	=	src/main.cpp 	\
		Tensor/MetaData.cpp 	\

OBJ	=	$(SRC:.cpp=.o)

CC	:=	g++
RM	:=	rm -rf
CPPFLAGS	=	-iquote Tensor/ -iquote Network/ -std=c++20 -g3
CFLAGS	=	-Wall

NAME	=	oceanTensor

all: $(NAME)

$(NAME):	$(OBJ)
	$(CC) -o $(NAME) $(OBJ) $(CFLAGS)

clean:
	$(RM) $(OBJ)

fclean: clean
	$(RM) $(NAME)

re: fclean all

.PHONY: fclean re all clean
