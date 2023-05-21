package main

import (
	"context"
	"fmt"
	"io"
	"os"
	"reflect"
	"unsafe"

	"github.com/JeremyLoy/config"
	openai "github.com/sashabaranov/go-openai"
)

type Env struct {
	OpenAIKey string `config:"OPENAI_KEY"`
}

func validate(e Env) error {
	t := reflect.TypeOf(e)
	var missing []string
	for i := 0; i < t.NumField(); i++ {
		f := t.Field(i)
		tag := f.Tag.Get("config")
		if tag == "" {
			continue
		}
		v := reflect.ValueOf(e).Field(i).String()
		if v == "" {
			missing = append(missing, tag)
		}
	}

	if len(missing) > 0 {
		return fmt.Errorf("required env variables: %v", missing)
	}

	return nil
}

func main() {
	e := Env{}
	err := config.FromEnv().To(&e)
	if err != nil {
		panic(err)
	}

	err = validate(e)
	if err != nil {
		panic(err)
	}

	c := New(e)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	if err := c.run(ctx); err != nil {
		panic(err)
	}
}

type client struct {
	Env

	oc *openai.Client
}

func New(e Env) *client {
	return &client{
		Env: e,
		oc:  openai.NewClient(e.OpenAIKey),
	}
}

func (c *client) run(ctx context.Context) error {
	b, err := io.ReadAll(os.Stdin)
	if err != nil {
		return fmt.Errorf("read from stdin: %w", err)
	}

	s := yoloString(b)

	res, err := c.gpt(ctx, s)
	if err != nil {
		return fmt.Errorf("gpt: %w", err)
	}

	_, err = fmt.Println(res)
	if err != nil {
		return fmt.Errorf("print result: %w", err)
	}

	return nil
}

func yoloString(b []byte) string {
	return *((*string)(unsafe.Pointer(&b)))
}

func (c *client) gpt(ctx context.Context, prompt string) (string, error) {
	res, err := c.oc.CreateChatCompletion(
		ctx,
		openai.ChatCompletionRequest{
			Model: openai.GPT3Dot5Turbo,
			Messages: []openai.ChatCompletionMessage{
				{
					Role:    openai.ChatMessageRoleUser,
					Content: prompt,
				},
			},
		},
	)

	if err != nil {
		return "", fmt.Errorf("create chat completion: %w", err)
	}

	return res.Choices[0].Message.Content, nil
}
