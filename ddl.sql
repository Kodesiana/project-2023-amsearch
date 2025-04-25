CREATE TABLE public.users (
	id varchar NOT NULL,
	username varchar NOT NULL,
	hashed_password varchar NOT NULL,
    CONSTRAINT users_pk PRIMARY KEY (id),
	CONSTRAINT users_username_unique UNIQUE (username)
);

INSERT INTO public.users (id, username, hashed_password) VALUES
('586061c5-8fe9-4e09-bd2c-47fc0ee9be35', 'admin', 'pbkdf2:sha256:600000$ULe3VYZ9Iygan53b$79fd5d79eae25fb9f61c14328eed082d1fe2f48f538c62fc8855babdaf181340');

CREATE TABLE public.documents (
	id varchar NOT NULL,
	title varchar NOT NULL,
    word_count integer NOT NULL,
	source_url varchar NULL,
	published_at timestamp with time zone NOT NULL,
	CONSTRAINT documents_pk PRIMARY KEY (id)
);

CREATE TABLE public.documents_raw (
	id varchar NOT NULL,
	parent_id varchar NOT NULL,
    title varchar NOT NULL,
	content text NOT NULL,
	embedding vector(768) NOT NULL,
    CONSTRAINT documents_raw_pk PRIMARY KEY (id),
	CONSTRAINT documents_raw_parent_fk FOREIGN KEY (parent_id) REFERENCES public.documents (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX documents_raw_search_idx ON documents_raw USING bm25 (id, title, content) WITH (key_field='id');

CREATE TABLE public.documents_stem (
	id varchar NOT NULL,
    parent_id varchar NOT NULL,
    title varchar NOT NULL,
	content text NOT NULL,
	embedding vector(768) NOT NULL,
	CONSTRAINT documents_stem_pk PRIMARY KEY (id),
	CONSTRAINT documents_stem_parent_fk FOREIGN KEY (parent_id) REFERENCES public.documents (id) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE INDEX documents_stem_search_idx ON documents_stem USING bm25 (id, title, content) WITH (key_field='id');

