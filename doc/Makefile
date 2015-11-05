include t2t_docsys/Makefile.include

.PHONY: all doc install
all: doc
doc: readme.txt readme.md
install: doc
	cp readme.txt ../README
	cp readme.md ../README.md

%.html: %.t2t $(wildcard *.t2t)
	$(call t2t_docsys_html,$@,$<)

%.txt: %.t2t $(wildcard *.t2t)
	$(call t2t_docsys_txt,$@,$<)

%.md: %.t2t $(wildcard *.t2t)
	$(call t2t_docsys_md,$@,$<)

.PHONY: clean
clean:
	@rm -f $(T2T_DOCSYS_OUTPUT_WILDCARDS)
